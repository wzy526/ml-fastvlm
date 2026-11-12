#!/usr/bin/env python3
"""Single-device Qwen2.5 learned-DAT integration check, no dataset needed.

Without head paths this constructs temporary random heads; generated text is
only a wiring check. --backward also checks answer-CE gradients to the sampler.
"""
import argparse
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from PIL import Image
import torch
from transformers import AutoConfig

from ldat_eval.infer import LDatHead, LDatRunner, global_positions
from llava.model.where_dat import _WhereXAttnHead


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_path", required=True)
    p.add_argument("--where_head_pt")
    p.add_argument("--ldat_pt")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--where_layer", type=int, default=8)
    p.add_argument("--lr_tok", type=int, default=64)
    p.add_argument("--crop_tok", type=int, default=16)
    p.add_argument("--hr_cap", type=int, default=256)
    p.add_argument("--backward", action="store_true")
    args = p.parse_args()
    cfg = AutoConfig.from_pretrained(args.model_path)
    if cfg.model_type != "qwen2_5_vl":
        raise ValueError(f"Expected native Qwen2.5-VL, got {cfg.model_type}")
    c = getattr(cfg, "text_config", cfg).hidden_size
    torch.manual_seed(42)
    with tempfile.TemporaryDirectory(prefix="qwen25-ldat-smoke-") as d:
        where = args.where_head_pt or str(Path(d) / "where.pt")
        ldat = args.ldat_pt or str(Path(d) / "ldat.pt")
        if not args.where_head_pt:
            torch.save(_WhereXAttnHead(c, c).state_dict(), where)
        if not args.ldat_pt:
            torch.save(LDatHead(c).state_dict(), ldat)
        runner = LDatRunner(args.model_path, where, ldat, args.device, model_family="qwen2_5_vl",
                            where_layer=args.where_layer, lr_tok=args.lr_tok, crop_tok=args.crop_tok,
                            hr_cap=args.hr_cap, fast_prefix=True)
        img = Image.fromarray(np.random.default_rng(42).integers(0, 256, (336, 504, 3), dtype=np.uint8))
        pr = runner.prepare(img, "Describe the image briefly.")
        if pr is None:
            raise RuntimeError("Failed to build the sampled image prompt")
        with torch.no_grad():
            output, _, _ = runner._prefill(pr)
        if not torch.isfinite(output.logits).all():
            raise RuntimeError("Non-finite prefill logits")
        print(f"[smoke] factor={runner.factor} prompt_tokens={pr['n_tokens']} "
              f"mosaic_tokens={sum(z.shape[0] for z in pr['feats'])}")
        print("[smoke] generated:", runner.generate(img, "Describe the image briefly.", 8))
        if args.backward:
            # prepare uses inference_mode; clone outside it before constructing
            # a differentiable sampled image and answer forward.
            wins = [{k: v.clone() if torch.is_tensor(v) else v for k, v in w.items()} for w in pr["wins"]]
            feats, points = zip(*(runner._ldat_tokens(w) for w in wins))
            emb = pr["emb"].clone()
            emb[0, pr["img_pos"].clone()] = torch.cat([pr["lr_emb"].clone(), *feats], 0).to(emb.dtype)
            pos = global_positions(pr["pos3d"].clone(), pr["blocks"], wins, points)
            result = runner.backbone.forward(inputs_embeds=emb, position_ids=pos,
                                              mm_token_type_ids=pr["mm"].clone(),
                                              use_cache=False, return_dict=True, logits_to_keep=1)
            target = runner.tok.encode("A", add_special_tokens=False)[0]
            loss = torch.nn.functional.cross_entropy(result.logits[:, -1].float(),
                                                      torch.tensor([target], device=args.device))
            loss.backward()
            grad = runner.head.out.weight.grad
            if grad is None or not torch.isfinite(grad).all() or not grad.abs().sum().item():
                raise RuntimeError("Answer CE did not reach the learned density head")
            if any(p.grad is not None for p in runner.model.parameters()):
                raise RuntimeError("Frozen backbone received parameter gradients")
            print(f"[smoke] CE={loss.item():.6f} density_grad={grad.norm().item():.6g}")
        print("[PASS] Qwen2.5 learned DAT wiring (not an accuracy benchmark)")


if __name__ == "__main__":
    main()
