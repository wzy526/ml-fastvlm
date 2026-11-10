"""lmms-eval model: Qwen3.5 with the finalized learnable DAT sampler.

Registered as ``qwen3_5_ldat`` through LMMS_EVAL_PLUGINS=ldat_eval.
The prompt the task builds is the question; this wrapper adds the LR image,
the where-head hint, and the learned mosaic, then generates greedily.
"""

from typing import List, Optional, Tuple

import torch
from accelerate import Accelerator
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms

from ldat_eval.infer import LDatRunner


class Qwen3_5_LDAT(lmms):
    def __init__(
        self,
        pretrained: str,
        where_head_pt: str,
        ldat_pt: str,
        crop_tok: int = 640,
        lr_tok: int = 640,
        dat_alpha: float = 0.8,
        dat_ratio: float = 0.0,
        dat_k: float = 0.6,
        src_scale: float = 4.0,
        ldat_d: int = 256,
        hr_cap: int = 4096,
        n_win: int = 1,
        grid: int = 0,
        batch_size: int = 1,
        attn_implementation: Optional[str] = "sdpa",
        enable_thinking: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"
        accelerator = Accelerator()
        self.accelerator = accelerator
        local = accelerator.local_process_index if accelerator.num_processes > 1 else 0
        if torch.cuda.is_available():
            torch.cuda.set_device(local)
            device = torch.device("cuda", local)
        else:
            device = torch.device("cpu")
        self._device = device
        self._rank = accelerator.local_process_index if accelerator.num_processes > 1 else 0
        self._world_size = accelerator.num_processes
        self.batch_size_per_gpu = int(batch_size)
        self._warned_multi = False
        self._printed_tok = False
        self.runner = LDatRunner(
            pretrained, where_head_pt, ldat_pt, device,
            crop_tok=crop_tok, lr_tok=lr_tok, dat_alpha=dat_alpha, dat_ratio=dat_ratio,
            dat_k=dat_k, src_scale=src_scale, ldat_d=ldat_d, hr_cap=hr_cap,
            n_win=n_win, grid=grid, attn=attn_implementation or "sdpa",
            enable_thinking=enable_thinking,
        )
        self._tokenizer = self.runner.tok
        self.pretrained = pretrained

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return 2048

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

    def _image_of(self, visual):
        if visual is None:
            return None
        if isinstance(visual, Image.Image):
            return visual
        if isinstance(visual, str):
            return Image.open(visual)
        if isinstance(visual, (list, tuple)):
            if len(visual) > 1 and not self._warned_multi:
                eval_logger.warning("ldat uses the first image of a multi-image sample")
                self._warned_multi = True
            return self._image_of(visual[0]) if visual else None
        return None

    def _clean(self, ans, until):
        if "</think>" in ans:
            ans = ans.partition("</think>")[2].strip()
        for term in until:
            if term:
                ans = ans.split(term)[0]
        return ans.strip()

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            return -len(self.tokenizer.encode(x[0])), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task, split = task[0], split[0]
            visuals = [doc_to_visual[i](self.task_dict[task][split][ids]) for i, ids in enumerate(doc_id)]
            gen_kwargs = all_gen_kwargs[0]
            until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])
            if isinstance(until, str):
                until = [until]
            until = [item for item in until if item != "\n\n"]
            max_new = int(gen_kwargs.get("max_new_tokens", 32))
            if not self._printed_tok:
                print(f"[ldat] enable_thinking={self.runner.enable_thinking} max_new_tokens={max_new}",
                      flush=True)
            for context, visual in zip(contexts, visuals):
                if isinstance(context, str) and "<image>" in context:
                    context = context.replace("<image>", "")
                image = self._image_of(visual)
                if image is None:
                    ans = ""
                else:
                    ans = self.runner.generate(image, context, max_new)
                    if not self._printed_tok and self.runner.last_n_tokens:
                        side = self.runner._mosaic_side()
                        wtxt = f"{self.runner.n_win}x{side}x{side}" if side else str(self.runner.crop_tok)
                        print(f"[ldat] prompt tokens {self.runner.last_n_tokens} "
                              f"(LR {self.runner.lr_tok} + window {wtxt} + text)", flush=True)
                        self._printed_tok = True
                res.append(self._clean(ans, until))
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), res[-1])
                pbar.update(1)
        res = re_ords.get_original(res)
        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("qwen3_5_ldat only generates")

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("qwen3_5_ldat has no multi-round generation")
