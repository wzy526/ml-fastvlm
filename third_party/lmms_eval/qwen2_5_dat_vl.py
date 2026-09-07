import base64
import math
import os
import re
import sys
from io import BytesIO
from typing import List, Optional, Tuple, Union

import decord
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.reasoning_model_utils import (
    parse_reasoning_model_answer,
)

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    eval_logger.warning(
        "Failed to import qwen_vl_utils. Please install via `pip install qwen-vl-utils`."
    )

def _fastvlm_candidates() -> List[str]:
    """Places ml-fastvlm may live, best guess first."""
    cands = [os.environ.get("LMMS_FASTVLM_PATH"), os.environ.get("FASTVLM_DIR")]
    # Normally a sibling of the lmms-eval checkout, or of the CWD.
    repo_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
    for base in (os.path.dirname(repo_root), os.getcwd(), os.path.dirname(os.getcwd())):
        cands.append(os.path.join(base, "ml-fastvlm"))
    # CFS mount points: pods/Host C vs the B-ws dev box.
    cands.append("/home/ea-cvfa-aigc-x2v-2/xzf/ml-fastvlm")
    cands.append("/media/cfs/ea-cvfa-aigc-x2v-2/xzf/ml-fastvlm")
    seen, out = set(), []
    for c in cands:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


_FASTVLM_CANDIDATES = _fastvlm_candidates()
_FASTVLM_PATH = next(
    (c for c in _FASTVLM_CANDIDATES if os.path.isdir(os.path.join(c, "llava"))),
    None,
)
if _FASTVLM_PATH and _FASTVLM_PATH not in sys.path:
    sys.path.insert(0, _FASTVLM_PATH)

try:
    # Training uses two-pass LSE attention in modeling_qwen2_5vl_dat.py (not manual).
    from llava.model.language_model.modeling_qwen2_5vl_dat import (
        Qwen2_5_VLDATConfig,
        Qwen2_5_VLDATForConditionalGeneration,
    )
    from transformers import AutoConfig, AutoModelForCausalLM
    AutoConfig.register("qwen2_5_vl_dat", Qwen2_5_VLDATConfig)
    AutoModelForCausalLM.register(Qwen2_5_VLDATConfig, Qwen2_5_VLDATForConditionalGeneration)
except ImportError as exc:
    raise ImportError(
        f"Cannot import Qwen2_5_VLDATForConditionalGeneration from ml-fastvlm "
        f"(tried path: {_FASTVLM_PATH!r}). "
        "Set LMMS_FASTVLM_PATH to the ml-fastvlm repo root. "
        "The LSE implementation also requires a working flash-attn stack "
        "(see ml-fastvlm _FA_BACKEND in modeling_qwen2_5vl_dat.py). "
        f"Original error: {exc}"
    ) from exc


@register_model("qwen2_5_dat_vl")
class Qwen2_5_DATVL(lmms):
    """
    Qwen2.5-VL with DAT (Dynamic Attention Token).
    Uses Qwen2_5_VLDATForConditionalGeneration from ml-fastvlm.

    In addition to standard Qwen2.5-VL parameters, the DAT variant processes
    each image twice: once at LR (standard resolution) and once at HR (high
    resolution). The HR features are injected into DAT attention layers.
    """

    # ---- family hooks (overridden by e.g. the Qwen3.5 DAT subclass) --------
    PATCH_SIZE = 14
    SPATIAL_MERGE = 2
    # Used only when a local DAT ckpt lacks preprocessor_config.json:
    # text hidden_size -> HF repo to source the image processor from.
    BASE_MODEL_MAP = {
        2048: "Qwen/Qwen2.5-VL-3B-Instruct",
        3584: "Qwen/Qwen2.5-VL-7B-Instruct",
        8192: "Qwen/Qwen2.5-VL-72B-Instruct",
    }
    DEFAULT_PROCESSOR_REF = "Qwen/Qwen2.5-VL-7B-Instruct"

    def _model_cls(self):
        return Qwen2_5_VLDATForConditionalGeneration

    def _process_vision_info(self, messages):
        return process_vision_info(messages)

    def _clean_answer(self, ans: str) -> str:
        return parse_reasoning_model_answer(ans)

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache: bool = True,
        attn_implementation: Optional[str] = None,
        min_pixels: int = 28224,
        max_pixels: int = 9031680,
        hr_scale: int = 3,
        use_lr_first_resize: bool = True,
        use_hr_first_resize: bool = False,
        use_decoupled_hr_lr: bool = False,
        lr_min_pixels: int = 200704,
        lr_max_pixels: int = 501760,
        hd_early_exit_k: int = 0,
        disable_hd: bool = False,
        max_num_frames: int = 32,
        use_custom_video_loader: Optional[bool] = False,
        fps: Optional[float] = None,
        max_image_size: Optional[int] = None,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        interleave_visuals: Optional[bool] = False,
        reasoning_prompt: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Qwen2.5-VL with DAT.

        Resolution semantics
        --------------------
        ``max_pixels`` / ``min_pixels`` always describe the **HR** (DAT input) range.
        ``lr_min_pixels`` / ``lr_max_pixels`` describe the LR (LLM-vision) range.

        Four modes (in priority order; first ``True`` flag wins):

        1. ``use_lr_first_resize=True`` (default, **mirrors training LR-first**):
            LR is the anchor. Per-image we run the LR ``image_processor`` first
            with band ``[lr_min_pixels, lr_max_pixels]`` to get ``lr_thw``, then
            derive
                ``hd_target = lr_pixels * hr_scale^2``
            capped at ``orig_pixels`` (and at ``max_pixels`` as an optional
            OOM guard). HR is then aspect-recovered with the **original** image
            aspect and snapped to ``FACTOR=14*2=28`` via floor:
                ``hd_h = max(28, (int(sqrt(hd_target/aspect)) // 28) * 28)``
            and the HR processor is forced to ``min=max=hd_total`` for that image.
            This is byte-for-byte the same geometry as
            ``Qwen2VLCoupledDATDataset`` legacy LR-first in
            ``ml-fastvlm/llava/train/train_qwen_dat.py``.

        2. ``use_hr_first_resize=True`` (deprecated):
            HR is the anchor; per-image LR_thw = floor(HR_thw / hr_scale) snapped
            to spatial_merge=2. Reverse direction of training -- kept for ablation.

        3. ``use_decoupled_hr_lr=True``:
            LR uses ``[lr_min_pixels, lr_max_pixels]`` independently of HR.
            HR uses ``[min_pixels, max_pixels]`` independently. LR/HR ratio is
            unconstrained per-image. Useful for HD-cap sweeps where LR is held
            constant.

        4. All False (legacy, old default before LR-first inference was added):
            LR uses [min_pixels // hr_scale^2, max_pixels // hr_scale^2];
            independent smart_resize within those bands.
        """
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        valid_attn = [None, "flash_attention_2", "sdpa", "eager"]
        if attn_implementation not in valid_attn:
            raise ValueError(
                f"attn_implementation must be one of {valid_attn}, got {attn_implementation}"
            )

        self.use_custom_video_loader = use_custom_video_loader
        self.fps = fps
        self.max_image_size = max_image_size
        if self.max_image_size and not self.use_custom_video_loader:
            raise ValueError("max_image_size is only applicable if use_custom_video_loader is True")

        accelerator = Accelerator()
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            # "auto" distributes layers across multiple GPUs which breaks HD feature
            # device placement in DAT model; fall back to the specified device.
            if device_map == "auto":
                self.device_map = device
            else:
                self.device_map = device_map if device_map else device

        model_kwargs: dict = {"torch_dtype": "bfloat16", "device_map": self.device_map}
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        self._model = self._model_cls().from_pretrained(
            pretrained, **model_kwargs
        ).eval()

        # HD ViT early exit: truncate the HD branch's ViT to the first k blocks
        # (0 = full depth). Overrides whatever the ckpt's config carries, so a
        # single ckpt can be swept over k from model_args without editing
        # config.json. NOTE: a ckpt trained at full depth will see a feature
        # distribution shift when k > 0 — accuracy deltas measure exactly that.
        if hd_early_exit_k:
            self._model.config.dat_extra_args["hd_early_exit_k"] = int(hd_early_exit_k)
            eval_logger.info(f"[DAT] hd_early_exit_k={hd_early_exit_k} (HD ViT truncated)")

        self.max_num_frames = max_num_frames

        # HR = user-specified max_pixels/min_pixels (DAT input resolution).
        # LR derivation depends on the mode (see docstring above).
        self.hr_max_pixels = max_pixels
        self.hr_min_pixels = min_pixels
        self.hr_scale = hr_scale
        # Ablation: run the DAT checkpoint with its HD branch switched off (LR
        # tokens only, pixel_values_hd=None). Isolates what the HD pathway adds
        # from what the DAT checkpoint's SFT data adds -- the vanilla baseline
        # never saw that data, so DAT-minus-vanilla conflates the two.
        self.disable_hd = bool(disable_hd)
        if self.disable_hd:
            eval_logger.info("[DAT] disable_hd=True: HD branch off, LR tokens only")
        self.use_lr_first_resize = use_lr_first_resize
        self.use_hr_first_resize = use_hr_first_resize
        self.use_decoupled_hr_lr = use_decoupled_hr_lr
        # Save explicit LR band; lr_first / decoupled rely on it directly.
        self.lr_min_pixels = lr_min_pixels
        self.lr_max_pixels = lr_max_pixels
        if use_lr_first_resize or use_decoupled_hr_lr:
            # LR is independent and base-VLM-compatible.
            self.max_pixels = lr_max_pixels
            self.min_pixels = lr_min_pixels
        else:
            # Legacy: LR derived from HR via hr_scale^2 ratio.
            self.max_pixels = max_pixels // (hr_scale ** 2)
            self.min_pixels = min_pixels // (hr_scale ** 2)
        # Geometry constants (family-dependent; see class attrs).
        self._patch_size = self.PATCH_SIZE
        self._spatial_merge = self.SPATIAL_MERGE
        self._factor = self._patch_size * self._spatial_merge  # 28 (qwen2.x) / 32 (qwen3.x): 1 merged token edge

        if reasoning_prompt:
            self.reasoning_prompt = reasoning_prompt.replace("\\n", "\n")
        else:
            self.reasoning_prompt = None

        # Resolve processor source: DAT checkpoints often lack preprocessor_config.json,
        # so fall back to the base Qwen2.5-VL model for the image processor.
        import os
        processor_path = pretrained
        if os.path.isdir(pretrained) and not os.path.exists(os.path.join(pretrained, "preprocessor_config.json")):
            hidden_size = self._model.config.text_config.hidden_size
            processor_path = self.BASE_MODEL_MAP.get(hidden_size, self.DEFAULT_PROCESSOR_REF)
            eval_logger.warning(
                f"No preprocessor_config.json in {pretrained}, "
                f"loading image processor from {processor_path}"
            )

        # ---- Processor construction-time bands ------------------------------
        # The chain at inference time is two-stage:
        #   stage 1: ``process_vision_info`` -> ``fetch_image`` smart_resizes
        #            each visual entry using its per-entry [min_pixels, max_pixels]
        #            (or the constructor band if absent).
        #   stage 2: ``self.processor(text=..., images=...)`` runs smart_resize
        #            AGAIN with the constructor band on the already-resized PIL.
        # To avoid stage 2 silently re-clipping the geometry that stage 1 just
        # locked in, the constructor band must be a superset of every per-image
        # target stage 1 can produce.
        _MERGED_TOKEN_PIXELS = self._factor * self._factor  # 784 = one merged token
        if use_lr_first_resize:
            # LR uses [lr_min, lr_max] both at stage 1 and stage 2.
            _lr_proc_min = self.lr_min_pixels
            _lr_proc_max = self.lr_max_pixels
            # HR is forced per-image to hd_total = (LR * hr_scale^2) capped by
            # orig_pixels (and optionally hr_max_pixels). hd_total can be smaller
            # than hr_min_pixels (tiny images) or larger than hr_max_pixels
            # (only if user lifted that cap). Open the constructor band wide.
            _hr_proc_min = _MERGED_TOKEN_PIXELS                   # 784
            _hr_proc_max = max(self.hr_max_pixels, 100_000_000)   # 100M ~ any real image
        elif use_hr_first_resize:
            _lr_proc_min = _MERGED_TOKEN_PIXELS                   # 784, per-image forces exact size
            _lr_proc_max = self.hr_max_pixels
            _hr_proc_min = self.hr_min_pixels
            _hr_proc_max = self.hr_max_pixels
        else:
            # Decoupled / legacy: each side uses its own band as stage-1 range.
            _lr_proc_min = self.min_pixels
            _lr_proc_max = self.max_pixels
            _hr_proc_min = self.hr_min_pixels
            _hr_proc_max = self.hr_max_pixels
        self.processor = AutoProcessor.from_pretrained(
            processor_path, max_pixels=_lr_proc_max, min_pixels=_lr_proc_min
        )
        self.hr_processor = AutoProcessor.from_pretrained(
            processor_path, max_pixels=_hr_proc_max, min_pixels=_hr_proc_min
        )

        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.system_prompt = system_prompt
        self.interleave_visuals = interleave_visuals
        # Extra kwargs forwarded to apply_chat_template (e.g. Qwen3.5's
        # enable_thinking). Subclasses populate this after super().__init__.
        self._template_kwargs: dict = {}

        self._config = self._model.config
        self._max_length = 2048
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self._model)
            else:
                self._model = accelerator.prepare_model(self._model, evaluation_mode=True)
            if accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = accelerator.local_process_index
            self._world_size = accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1
            

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("loglikelihood is not implemented for Qwen2_5_DATVL")

    def flatten(self, input: list) -> list:
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL image to base64-encoded **PNG** (lossless) data URI.

        Switched from JPEG (default quality=75) to PNG so that eval-time image
        decoding matches training fidelity. Training reads the source file with
        ``Image.open`` and feeds the PIL object directly to the processor (see
        ``Qwen2VLCoupledDATDataset._get_item`` in
        ``ml-fastvlm/llava/train/train_qwen_dat.py``); a JPEG re-encode on the
        eval side would inject DCT-domain noise that training never sees and
        is especially harmful for fine-detail benchmarks (vstar, ocrbench).
        ``qwen_vl_utils.fetch_image`` accepts any PIL-readable format.
        """
        rgb = image.convert("RGB")
        buffer = BytesIO()
        rgb.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    def _build_visual_entries(self, visual_list: list, max_pixels: int, min_pixels: int) -> list:
        """Build processed visual entries (image/video dicts) for the given pixel limits."""
        processed = []
        for visual in visual_list:
            if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):
                processed.append(
                    {
                        "type": "video",
                        "video": visual,
                        "max_pixels": max_pixels,
                        "min_pixels": min_pixels,
                    }
                )
            elif isinstance(visual, Image.Image):
                processed.append(
                    {
                        "type": "image",
                        "image": self._image_to_base64(visual),
                        "max_pixels": max_pixels,
                        "min_pixels": min_pixels,
                    }
                )
        return processed

    def _derive_lr_pixels_from_hr(self, image: Image.Image) -> int:
        """Per-image LR pixel target = floor(HR // hr_scale)^2 in 14-px patches.

        Runs HR image_processor once to get the smart-resized ``image_grid_thw``,
        then derives LR (h, w) in 14-px patches snapped to ``spatial_merge=2``.

        The returned value is exactly ``lr_h * lr_w * 14^2`` and is a multiple
        of ``(spatial_merge * 14)^2 = 28^2`` so the LR smart_resize is exact.
        """
        hr_inp = self.hr_processor.image_processor(
            images=[image.convert("RGB")], return_tensors="pt"
        )
        hd_thw = hr_inp["image_grid_thw"][0]
        hd_h_patches = int(hd_thw[1].item())
        hd_w_patches = int(hd_thw[2].item())
        sm = self._spatial_merge
        lr_h_patches = max(sm, (hd_h_patches // self.hr_scale // sm) * sm)
        lr_w_patches = max(sm, (hd_w_patches // self.hr_scale // sm) * sm)
        return lr_h_patches * lr_w_patches * self._patch_size * self._patch_size

    def _derive_hr_size_from_lr_first(self, image: Image.Image) -> Tuple[int, int]:
        """Per-image HR (h, w) target = (LR_pixels * hr_scale^2) aspect-snapped.

        Mirror image of the training-time LR-first geometry in
        ``Qwen2VLCoupledDATDataset._get_item`` (``ml-fastvlm``):

            1. Run LR ``image_processor`` with [lr_min, lr_max] to get
               ``lr_thw``; then ``lr_pixels = lr_h * lr_w * 14^2``.
            2. ``hd_target = lr_pixels * hr_scale^2``,
               clamped by ``orig_pixels`` (image cannot be upscaled past itself)
               and by ``hr_max_pixels`` (optional OOM guard; default 9031680
               which is wider than training's 3211264 so usually a no-op).
            3. Aspect-recover with the **original** image's aspect ratio
               (NOT the LR_thw aspect) and floor-snap each dim to FACTOR=28:
                   ``hd_h = max(28, (int(sqrt(hd_target/aspect)) // 28) * 28)``
            4. Return ``(hd_h, hd_w)``. The caller passes these as
               ``resized_height/resized_width`` so fetch_image resizes to this
               exact geometry.

        NOTE: this used to return ``hd_h * hd_w`` for a forced
        ``min=max=hd_total`` band, but qwen_vl_utils' smart_resize downscale
        branch has no ``max(factor, ...)`` guard: for extreme-aspect images
        (e.g. 3000x20) the per-dim ``max(factor, ...)`` clamps here inflate
        hd_total slightly above what the original dims can express, smart_resize
        then "downscales" and floors the short side to 0 ->
        ``ValueError: height and width must be > 0`` in PIL (observed on gen7
        tok256; the dead rank left the others waiting in the final gather).
        The ``resized_height/width`` path only factor-snaps with a
        ``max(factor, ...)`` guard, so it cannot produce 0 and is geometry-exact.

        Returns the per-image ``(hd_h, hd_w)`` in pixels.
        """
        lr_inp = self.processor.image_processor(
            images=[image.convert("RGB")], return_tensors="pt",
        )
        lr_thw = lr_inp["image_grid_thw"][0]
        # image_grid_thw counts UNMERGED patches, so the edge scale is
        # patch_size, not _factor (= patch_size * spatial_merge). Using _factor
        # here overestimated lr_pixels 4x, which turned hd_target into 36x the
        # LR area instead of hr_scale^2 = 9x and pinned every sweep point to
        # hr_max_pixels.
        lr_h_px = int(lr_thw[1].item()) * self._patch_size
        lr_w_px = int(lr_thw[2].item()) * self._patch_size
        lr_pixels = lr_h_px * lr_w_px

        orig_pixels = image.width * image.height
        hd_target = lr_pixels * (self.hr_scale ** 2)
        hd_target = min(hd_target, orig_pixels)
        hd_target = min(hd_target, self.hr_max_pixels)

        aspect = image.width / image.height
        hd_h = int(math.sqrt(hd_target / aspect))
        hd_w = int(hd_h * aspect)
        hd_h = max(self._factor, (hd_h // self._factor) * self._factor)
        hd_w = max(self._factor, (hd_w // self._factor) * self._factor)
        return hd_h, hd_w

    def _build_visual_entries_lr_first(self, visual_list: list):
        """LR-first variant: per-image HR pixel target derived from LR thw.

        Returns (lr_visuals, hr_visuals). The LR entry uses the
        [lr_min_pixels, lr_max_pixels] band (so LR smart_resize matches
        training's ``processor(min_pixels=lr_min, max_pixels=lr_max)`` call);
        the HR entry is forced to ``min=max=hd_total`` per-image.

        Videos fall back to independent resize because per-frame DAT geometry
        alignment is not yet supported here.
        """
        lr_visuals: list = []
        hr_visuals: list = []
        for visual in visual_list:
            if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):
                lr_visuals.append({
                    "type": "video", "video": visual,
                    "max_pixels": self.lr_max_pixels, "min_pixels": self.lr_min_pixels,
                })
                hr_visuals.append({
                    "type": "video", "video": visual,
                    "max_pixels": self.hr_max_pixels, "min_pixels": self.hr_min_pixels,
                })
            elif isinstance(visual, Image.Image):
                hd_h, hd_w = self._derive_hr_size_from_lr_first(visual)
                # Pass PIL directly (qwen_vl_utils.fetch_image accepts PIL).
                # No JPEG round-trip => bit-exact match to training fidelity.
                lr_visuals.append({
                    "type": "image", "image": visual,
                    "max_pixels": self.lr_max_pixels, "min_pixels": self.lr_min_pixels,
                })
                # resized_height/width (NOT a forced min=max pixel band): see
                # _derive_hr_size_from_lr_first — the band form crashes
                # qwen_vl_utils' unguarded floor on extreme-aspect images.
                hr_visuals.append({
                    "type": "image", "image": visual,
                    "resized_height": hd_h, "resized_width": hd_w,
                })
        return lr_visuals, hr_visuals

    def _build_visual_entries_hr_first(self, visual_list: list):
        """HR-first variant: per-image LR pixel target derived from HR thw.

        Returns (lr_visuals, hr_visuals). Videos fall back to the legacy
        independent-resize path because per-frame DAT geometry alignment is
        not yet supported here.
        """
        lr_visuals: list = []
        hr_visuals: list = []
        for visual in visual_list:
            if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):
                # Videos: fall back to independent resize (no per-frame HR alignment yet)
                video_entry_lr = {
                    "type": "video",
                    "video": visual,
                    "max_pixels": self.max_pixels,
                    "min_pixels": self.min_pixels,
                }
                video_entry_hr = {
                    "type": "video",
                    "video": visual,
                    "max_pixels": self.hr_max_pixels,
                    "min_pixels": self.hr_min_pixels,
                }
                lr_visuals.append(video_entry_lr)
                hr_visuals.append(video_entry_hr)
            elif isinstance(visual, Image.Image):
                lr_pixels = self._derive_lr_pixels_from_hr(visual)
                b64 = self._image_to_base64(visual)
                lr_visuals.append(
                    {
                        "type": "image",
                        "image": b64,
                        "max_pixels": lr_pixels,
                        "min_pixels": lr_pixels,
                    }
                )
                hr_visuals.append(
                    {
                        "type": "image",
                        "image": b64,
                        "max_pixels": self.hr_max_pixels,
                        "min_pixels": self.hr_min_pixels,
                    }
                )
        return lr_visuals, hr_visuals

    def _sample_video_frames(self, video_inputs):
        """Sub-sample video frames to max_num_frames."""
        if video_inputs is None:
            return video_inputs
        total_frames = video_inputs[0].shape[0]
        indices = np.linspace(0, total_frames - 1, self.max_num_frames, dtype=int)
        indices = np.unique(indices)
        if total_frames - 1 not in indices:
            indices = np.append(indices, total_frames - 1)
            indices = np.unique(indices)
        video_inputs[0] = video_inputs[0][indices]
        return video_inputs

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            try:
                task = task[0]
                split = split[0]
                visual_list = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
                gen_kwargs = all_gen_kwargs[0]

                until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(
                        f"Expected `gen_kwargs['until']` to be Union[str, list], got {type(until)}"
                    )
                # Avoid '\n\n' as a stopper to prevent premature truncation
                until = [item for item in until if item != "\n\n"]

                if isinstance(contexts, tuple):
                    contexts = list(contexts)
                for i in range(len(contexts)):
                    if "<image>" in contexts[i]:
                        contexts[i] = contexts[i].replace("<image>", "")

                batched_messages = []
                batched_messages_hr = []  # HR messages for DAT HD features

                for i, context in enumerate(contexts):
                    if self.reasoning_prompt:
                        context = context.strip() + self.reasoning_prompt
                        contexts[i] = context

                    message = [{"role": "system", "content": self.system_prompt}]

                    if self.use_lr_first_resize:
                        # LR-first (default): mirrors training Qwen2VLCoupledDATDataset
                        # legacy LR-first geometry. LR independent in [lr_min, lr_max];
                        # HR forced to (LR * hr_scale^2) aspect-snapped per-image.
                        lr_visuals, hr_visuals = self._build_visual_entries_lr_first(
                            visual_list[i] or []
                        )
                    elif self.use_hr_first_resize:
                        # [Deprecated] HR-first: per-image LR pixel target = floor(HR/hr_scale)^2.
                        lr_visuals, hr_visuals = self._build_visual_entries_hr_first(
                            visual_list[i] or []
                        )
                    else:
                        # Decoupled or legacy: LR / HR independently smart-resized
                        # within their own [min, max] bands. self.max_pixels / self.min_pixels
                        # is set in __init__ based on use_decoupled_hr_lr.
                        lr_visuals = self._build_visual_entries(
                            visual_list[i] or [], self.max_pixels, self.min_pixels
                        )
                        hr_visuals = self._build_visual_entries(
                            visual_list[i] or [], self.hr_max_pixels, self.hr_min_pixels
                        )

                    if not self.interleave_visuals:
                        message.append(
                            {
                                "role": "user",
                                "content": lr_visuals + [{"type": "text", "text": context}],
                            }
                        )
                    else:
                        image_placeholders = re.findall(r"<image \d+>", context)
                        content_parts = []
                        text_parts = re.split(r"<image \d+>", context)
                        if text_parts[0]:
                            content_parts.append({"type": "text", "text": text_parts[0]})
                        for j, placeholder in enumerate(image_placeholders):
                            match = re.search(r"<image (\d+)>", placeholder)
                            img_idx = int(match.group(1)) - 1 if match else 0
                            image_idx = min(img_idx, len(lr_visuals) - 1) if lr_visuals else 0
                            if lr_visuals and image_idx < len(lr_visuals):
                                content_parts.append(lr_visuals[image_idx])
                            if j + 1 < len(text_parts) and text_parts[j + 1]:
                                content_parts.append({"type": "text", "text": text_parts[j + 1]})
                        message.append({"role": "user", "content": content_parts})

                    batched_messages.append(message)

                    # Build HR messages (same structure but with HR visual entries)
                    message_hr = [{"role": "system", "content": self.system_prompt}]
                    message_hr.append(
                        {
                            "role": "user",
                            "content": hr_visuals + [{"type": "text", "text": context}],
                        }
                    )
                    batched_messages_hr.append(message_hr)

                # LR inputs (standard Qwen2.5-VL inputs)
                texts = [
                    self.processor.apply_chat_template(
                        msg, tokenize=False, add_generation_prompt=True, **self._template_kwargs
                    )
                    for msg in batched_messages
                ]
                image_inputs, video_inputs = self._process_vision_info(batched_messages)
                if video_inputs is not None:
                    video_inputs = self._sample_video_frames(video_inputs)
                inputs = self.processor(
                    text=texts,
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )

                # HR inputs for DAT HD features (images only, no text needed)
                hr_image_inputs, _ = (
                    (None, None) if self.disable_hd
                    else self._process_vision_info(batched_messages_hr)
                )
                pixel_values_hd = None
                image_grid_thw_hd = None
                if hr_image_inputs is not None and len(hr_image_inputs) > 0:
                    # Use HR processor to get pixel_values_hd and image_grid_thw_hd
                    hr_texts = [
                        self.hr_processor.apply_chat_template(
                            msg, tokenize=False, add_generation_prompt=True, **self._template_kwargs
                        )
                        for msg in batched_messages_hr
                    ]
                    hr_inputs = self.hr_processor(
                        text=hr_texts,
                        images=hr_image_inputs,
                        padding=True,
                        return_tensors="pt",
                    )
                    pixel_values_hd = hr_inputs.get("pixel_values")
                    image_grid_thw_hd = hr_inputs.get("image_grid_thw")

                # Move inputs to device
                if self.device_map == "auto":
                    target_device = "cuda"
                else:
                    target_device = self.device
                inputs = inputs.to(target_device)
                if pixel_values_hd is not None:
                    pixel_values_hd = pixel_values_hd.to(target_device)
                if image_grid_thw_hd is not None:
                    image_grid_thw_hd = image_grid_thw_hd.to(target_device)

                # Generation config
                default_gen_kwargs = {
                    "max_new_tokens": 32768,
                    "temperature": 0.0,
                    "top_p": None,
                    "num_beams": 1,
                }
                current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
                pad_token_id = self.tokenizer.pad_token_id

                if current_gen_kwargs["temperature"] > 0:
                    current_gen_kwargs["do_sample"] = True
                else:
                    current_gen_kwargs["do_sample"] = False
                    current_gen_kwargs["temperature"] = None
                    current_gen_kwargs["top_p"] = None

                # Keys accepted by Qwen2.5-VL generate; drop extras like mm_token_type_ids
                # that the fast image processor may inject but the model does not accept.
                _GENERATE_INPUT_KEYS = {
                    "input_ids", "attention_mask", "pixel_values",
                    "image_grid_thw", "video_grid_thw", "rope_deltas",
                }
                generate_kwargs = {
                    **{k: v for k, v in inputs.items() if k in _GENERATE_INPUT_KEYS},
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "pad_token_id": pad_token_id,
                    "do_sample": current_gen_kwargs["do_sample"],
                    "temperature": current_gen_kwargs["temperature"],
                    "top_p": current_gen_kwargs["top_p"],
                    "num_beams": current_gen_kwargs["num_beams"],
                    "max_new_tokens": current_gen_kwargs["max_new_tokens"],
                    "use_cache": self.use_cache,
                }
                # Pass DAT HD features when available
                if pixel_values_hd is not None:
                    generate_kwargs["pixel_values_hd"] = pixel_values_hd
                if image_grid_thw_hd is not None:
                    generate_kwargs["image_grid_thw_hd"] = image_grid_thw_hd
            
                cont = self.model.generate(**generate_kwargs)

                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)
                ]
                answers = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

                for i, ans in enumerate(answers):
                    for term in until:
                        if len(term) > 0:
                            ans = ans.split(term)[0]
                    answers[i] = ans

                for ans, context in zip(answers, contexts):
                    clean_ans = self._clean_answer(ans)
                    res.append(clean_ans)
                    self.cache_hook.add_partial("generate_until", (context, gen_kwargs), clean_ans)
                    pbar.update(1)
            except Exception as exc:
                # One bad sample must not kill the rank: lmms-eval catches the
                # exception at cli level and the dead rank then leaves every
                # other rank waiting forever in the final gather (observed:
                # extreme-aspect image -> ValueError in fetch_image -> 6h
                # distributed deadlock). Degrade to empty answers instead so
                # all ranks stay in lockstep; the samples just score 0.
                eval_logger.error(
                    f"generate_until failed on task={task} doc_id={list(doc_id)}: "
                    f"{type(exc).__name__}: {exc} — returning empty answers for this chunk"
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                _gk = all_gen_kwargs[0]
                for _ctx in contexts:
                    res.append("")
                    self.cache_hook.add_partial("generate_until", (_ctx, _gk), "")
                    pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("generate_until_multi_round is not implemented for Qwen2_5_DATVL")
