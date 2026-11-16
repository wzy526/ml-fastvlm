"""Methods for scripts/bench_cost.py.

Each class loads one method and answers one (image, prompt) pair with
``run(image, question, max_new_tokens, clock)``. ``run`` calls
``clock.mark("prepare")`` once the model input is ready, ``clock.mark("prefill")``
after the forward that yields the first token, and ``clock.mark("decode")``
after the last token. Decoding is greedy and stops at an end token or at
``max_new_tokens``. ``groups`` names the modules whose GPU time and FLOPs are
reported separately; groups must not contain each other.

New methods: a class with the same ``__init__(device=..., attn=..., **model_args)``
/ ``groups`` / ``run`` shape, registered in MODELS or passed as
``--model package.module:Class``.
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)


def _eos_ids(model, tok):
    eos = set()
    for v in (model.generation_config.eos_token_id, tok.eos_token_id,
              tok.convert_tokens_to_ids("<|im_end|>")):
        if isinstance(v, (list, tuple)):
            eos.update(int(x) for x in v if x is not None)
        elif v is not None:
            eos.add(int(v))
    return eos


class QwenNative:
    """Stock Qwen VLM (Qwen3.5, Qwen2.5-VL, Qwen3-VL) on one image.

    max_tok > 0: the processor caps the image at that many visual tokens
    (the LR rows). max_tok = 0: the checkpoint's own processor limits.
    max_tok < 0: no cap, the image keeps its resolution (an 8K frame is
    ~37k tokens). Prompt = lmms-eval's: system line, image, task text.
    """

    def __init__(self, pretrained, max_tok=640, enable_thinking=False,
                 system_prompt="You are a helpful assistant.", device="cuda", attn="sdpa"):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.device = device
        self.model = AutoModelForImageTextToText.from_pretrained(
            pretrained, torch_dtype=torch.bfloat16, device_map={"": device},
            attn_implementation=attn).eval()
        vis = self.model.config.vision_config
        factor = int(vis.patch_size) * int(vis.spatial_merge_size)
        max_tok = int(max_tok)
        if max_tok > 0:
            kw = dict(min_pixels=factor ** 2, max_pixels=max_tok * factor ** 2)
        elif max_tok < 0:
            kw = dict(min_pixels=factor ** 2, max_pixels=100_000_000)
        else:
            kw = {}
        self.processor = AutoProcessor.from_pretrained(pretrained, **kw)
        self.tok = self.processor.tokenizer
        self.enable_thinking = bool(enable_thinking)
        self.system_prompt = system_prompt
        self.image_id = self.model.config.image_token_id
        self.eos = _eos_ids(self.model, self.tok)
        inner = self.model.model
        self.groups = {"visual": inner.visual, "language_model": inner.language_model,
                       "lm_head": self.model.lm_head}

    @torch.inference_mode()
    def run(self, image, question, max_new_tokens, clock):
        from bench_ldat_speed import _text_position

        dev = self.device
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": [{"type": "image", "image": image},
                                                     {"type": "text", "text": question}]})
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.enable_thinking)
        enc = self.processor(text=[text], images=[image], return_tensors="pt")
        inp = {k: (v.to(dev, dtype=torch.bfloat16) if k == "pixel_values" else v.to(dev))
               for k, v in enc.items() if torch.is_tensor(v)}
        clock.mark("prepare")
        out = self.model(**inp, use_cache=True, return_dict=True, logits_to_keep=1)
        clock.mark("prefill")
        n = int(inp["input_ids"].shape[1])
        rope_deltas = getattr(self.model.model, "rope_deltas", None)
        cache = out.past_key_values
        toks = []
        tok = int(out.logits[0, -1].argmax())
        while True:
            toks.append(tok)
            if tok in self.eos or len(toks) >= max_new_tokens:
                break
            cp = torch.tensor([n + len(toks) - 1], device=dev)
            out = self.model(input_ids=torch.tensor([[tok]], device=dev),
                             position_ids=_text_position(cp, rope_deltas, dev),
                             past_key_values=cache, cache_position=cp,
                             use_cache=True, return_dict=True, logits_to_keep=1)
            cache = out.past_key_values
            tok = int(out.logits[0, -1].argmax())
        clock.mark("decode")
        return dict(visual_tokens=int((inp["input_ids"][0] == self.image_id).sum()),
                    prompt_tokens=n, gen_tokens=len(toks),
                    text=self.tok.decode(toks, skip_special_tokens=True))


class LDat:
    """Q-LAS through ldat_eval.LDatRunner: the same prepare / prefill / decode
    as the lmms-eval plugin. "prepare" covers the LR first pass, where-head,
    window ViT, sampling head and mosaic tokens."""

    DEFAULTS = dict(lr_tok=640, crop_tok=400, n_win=1, grid=0, dat_alpha=0.8, dat_ratio=0.0,
                    dat_k=0.6, src_scale=4.0, supersample=2, ldat_d=256, hr_cap=4096,
                    where_layer=8, fast_prefix=False, sample_feat=False, prefix_cache=False,
                    enable_thinking=False, win_full=False)

    def __init__(self, pretrained, where_head_pt, ldat_pt, family="auto", device="cuda",
                 attn="sdpa", **kw):
        from ldat_eval.infer import LDatRunner

        unknown = set(kw) - set(self.DEFAULTS)
        if unknown:
            raise ValueError(f"ldat: unknown model_args {sorted(unknown)}")
        params = {**self.DEFAULTS, **kw}
        for key in ("fast_prefix", "sample_feat", "prefix_cache", "enable_thinking", "win_full"):
            params[key] = _as_bool(params[key])
        self.params = params
        self.runner = LDatRunner(pretrained, where_head_pt, ldat_pt, device,
                                 model_family=family, attn=attn, **params)
        r = self.runner
        self.groups = {"visual": r.model.model.visual, "language_model": r.model.model.language_model,
                       "lm_head": r.model.lm_head, "where_head": r.model.where_head, "ldat_head": r.head}

    @torch.inference_mode()
    def run(self, image, question, max_new_tokens, clock):
        r = self.runner
        pr = r.prepare(image, question)
        clock.mark("prepare")
        if pr is None:
            return None
        o, pos, n = r._prefill(pr)
        clock.mark("prefill")
        text = r._decode(o, pos, n, max_new_tokens, None, pr["ids"].tolist())
        clock.mark("decode")
        n_lr = int(torch.prod(pr["lr_thw"][0])) // (r.ms * r.ms)
        n_win = int(sum(w["gh"] * w["gw"] for w in pr["wins"]))
        return dict(visual_tokens=n_lr + n_win, lr_tokens=n_lr, win_tokens=n_win,
                    prompt_tokens=int(pr["n_tokens"]),
                    gen_tokens=int(r.last_generation["output_tokens"]), text=text)


def _as_bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes"}


MODELS = {"qwen_native": QwenNative, "ldat": LDat}
