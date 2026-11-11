"""Answer one image with the finalized learnable sampler.

Same path as scripts/offset_feat_train.py accuracy mode for ``ldat``:
layer-8 where-head, analytic adaptive window, LDatHead on the window's
native ViT map, inverse-CDF mosaic, global M-RoPE. No window-scale head.
"""

import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import llava.model.language_model.modeling_qwen3_5_dat as dat_mod
from offset_feat_train import LDatHead, global_positions, inv_cdf, nms_peaks
from patch_sample_probe import crop_geometry


class LDatRunner:
    def __init__(
        self,
        model_path,
        where_head_pt,
        ldat_pt,
        device,
        crop_tok=640,
        lr_tok=640,
        dat_alpha=0.8,
        dat_ratio=0.0,
        dat_k=0.6,
        src_scale=4.0,
        supersample=2,
        ldat_d=256,
        hr_cap=4096,
        where_layer=8,
        n_win=1,
        grid=0,
        attn="sdpa",
        enable_thinking=False,
        prefix_cache=False,
    ):
        self.crop_tok = int(crop_tok)
        self.lr_tok = int(lr_tok)
        self.n_win = max(int(n_win), 1)
        # grid>0 forces each window to N x N. Several windows with grid left at 0
        # use 20 x 20 each, so the extra windows add tokens instead of splitting 400.
        self.grid = int(grid)
        self.dat_alpha = float(dat_alpha)
        self.dat_ratio = float(dat_ratio)
        self.dat_k = float(dat_k)
        self.src_scale = float(src_scale)
        self.supersample = int(supersample)
        self.hr_cap = int(hr_cap)
        self.where_layer = int(where_layer)
        self.enable_thinking = bool(enable_thinking)
        self.prefix_cache = bool(prefix_cache)
        self.dev = device

        from transformers import AutoProcessor

        self.model = dat_mod.Qwen3_5DATForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map={"": device}, attn_implementation=attn
        ).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._load_where_head(os.path.expanduser(where_head_pt))

        self.processor = AutoProcessor.from_pretrained(model_path, min_pixels=1024, max_pixels=self.lr_tok * 1024)
        self.hr_processor = AutoProcessor.from_pretrained(model_path, min_pixels=1024, max_pixels=100_000_000)
        self.tok = self.hr_processor.tokenizer
        ip = self.hr_processor.image_processor
        self.ps, self.tps, self.ms = ip.patch_size, ip.temporal_patch_size, ip.merge_size
        self.mean = torch.tensor(ip.image_mean, device=device).view(3, 1, 1)
        self.std = torch.tensor(ip.image_std, device=device).view(3, 1, 1)
        self.image_id = self.model.config.image_token_id
        c = self.model.config.text_config.hidden_size
        self.embed = self.model.get_input_embeddings()
        self.head = LDatHead(c, d=int(ldat_d)).to(device).eval()
        self._load_ldat(os.path.expanduser(ldat_pt))
        eos = set()
        for v in (self.model.generation_config.eos_token_id, self.tok.eos_token_id,
                  self.tok.convert_tokens_to_ids("<|im_end|>")):
            if isinstance(v, (list, tuple)):
                eos.update(int(x) for x in v if x is not None)
            elif v is not None:
                eos.add(int(v))
        self.eos = eos
        self.think_end = None
        if self.enable_thinking:
            tid = self.tok.convert_tokens_to_ids("</think>")
            unk = self.tok.unk_token_id
            if isinstance(tid, int) and tid >= 0 and (unk is None or tid != int(unk)):
                self.think_end = int(tid)
        self.n_think_closed = 0
        self.n_think_capped = 0
        self.n_cache = 0
        self.n_cache_fallback = 0
        self.last_n_tokens = 0
        w = float(self.head.out.weight.detach().abs().mean())
        side = self._mosaic_side()
        geom = f"{self.n_win} x {side}x{side}" if side else f"1 x crop_tok {self.crop_tok}"
        print(f"[ldat] windows {geom}  lr_tok {self.lr_tok}  alpha {self.dat_alpha}  "
              f"ratio {self.dat_ratio}  head |w| {w:.3e}", flush=True)

    def _mosaic_side(self):
        if self.grid > 0:
            return self.grid
        if self.n_win > 1:
            return 20
        return 0

    def _load_where_head(self, path):
        sd = torch.load(path, map_location="cpu")
        sd = {k[len("where_head."):] if k.startswith("where_head.") else k: v for k, v in sd.items()}
        if not hasattr(self.model, "where_head"):
            hidden = self.model.config.text_config.hidden_size
            self.model.where_head = dat_mod._WhereXAttnHead(hidden, hidden, d=384, layers=2, heads=6, max_hw=512)
        self.model.where_head.load_state_dict(sd)
        self.model.where_head.float().to(self.dev).eval()

    def _load_ldat(self, path):
        sd = torch.load(path, map_location="cpu")
        state = sd["head"] if isinstance(sd, dict) and "head" in sd else sd
        self.head.load_state_dict(state)
        if isinstance(sd, dict) and sd.get("win_head"):
            print("[ldat] checkpoint has a window head; this eval does not use it", flush=True)

    def _patchify(self, x):
        _, h, w = x.shape
        x = ((x - self.mean) / self.std)[None].repeat(self.tps, 1, 1, 1)
        gh, gw = h // self.ps, w // self.ps
        x = x.reshape(1, self.tps, 3, gh // self.ms, self.ms, self.ps, gw // self.ms, self.ms, self.ps)
        x = x.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        return x.reshape(gh * gw, 3 * self.tps * self.ps * self.ps)

    def _vit(self, pv, thw):
        e = self.model.model.visual(pv.to(torch.bfloat16), grid_thw=thw, return_dict=True).pooler_output
        return torch.cat(list(e), 0) if isinstance(e, (list, tuple)) else e

    def _vit_image(self, pil):
        x = torch.from_numpy(np.asarray(pil, dtype=np.float32) / 255.0).permute(2, 0, 1).to(self.dev)
        h, w = x.shape[1] // self.ps, x.shape[2] // self.ps
        e = self._vit(self._patchify(x), torch.tensor([[1, h, w]], device=self.dev))
        return e.reshape(h // self.ms, w // self.ms, -1)

    def _axis_density(self, m, g=64):
        m = F.interpolate(m[None, None], size=g, mode="linear", align_corners=False)[0, 0].clamp_min(0)
        d = (1 - self.dat_alpha) / g + self.dat_alpha * m / m.sum().clamp_min(1e-8)
        if self.dat_ratio > 1:
            d = torch.maximum(d, d.max() / self.dat_ratio)
        return d / d.sum()

    def _ldat_tokens(self, w):
        """Window features + where-head log p -> mosaic tokens and global-PE points."""
        x0, y0, x1, y1 = w["lnbox"]
        gh, gw, s = w["gh"], w["gw"], max(self.supersample, 1)
        nh, nw = w["lmap"].shape[-2:]
        uh = (torch.arange(nh, device=self.dev) + 0.5) / nh
        uw = (torch.arange(nw, device=self.dev) + 0.5) / nw
        gy, gx = torch.meshgrid((y0 + uh * (y1 - y0)) * 2 - 1, (x0 + uw * (x1 - x0)) * 2 - 1, indexing="ij")
        gc = torch.stack([gx, gy], -1)[None]
        f = w["lmap"]
        pw = F.grid_sample(w["head_p"][None, None], gc, mode="bilinear", padding_mode="border",
                           align_corners=False)[0, 0].clamp_min(1e-8)
        lp = pw.log()
        q = torch.softmax((lp + self.head(f, lp)).flatten(), 0).reshape(nh, nw)
        dx, dy = self._axis_density(q.sum(0)), self._axis_density(q.sum(1))
        ux = (torch.arange(gw * 32 * s, device=self.dev) + 0.5) / (gw * 32 * s)
        uy = (torch.arange(gh * 32 * s, device=self.dev) + 0.5) / (gh * 32 * s)
        sy, sx = torch.meshgrid(inv_cdf(dy, uy) * 2 - 1, inv_cdf(dx, ux) * 2 - 1, indexing="ij")
        img = F.grid_sample(w["psrc"][None], torch.stack([sx, sy], -1)[None], mode="bilinear",
                            padding_mode="border", align_corners=False)
        if img.shape[-2:] != (gh * 32, gw * 32):
            img = F.interpolate(img, size=(gh * 32, gw * 32), mode="bicubic", antialias=True, align_corners=False)
        img = img[0].clamp(0, 1)
        z = self._vit(self._patchify(img), torch.tensor([[1, gh * 32 // self.ps, gw * 32 // self.ps]], device=self.dev))
        z = z.reshape(gh * gw, -1)
        lx0, ly0, lx1, ly1 = w["lpx"]
        cx0, cy0, cx1, cy1 = w["px"]
        cy = (ly0 + inv_cdf(dy, (torch.arange(gh, device=self.dev) + 0.5) / gh) * (ly1 - ly0) - cy0) / (cy1 - cy0) * 2 - 1
        cx = (lx0 + inv_cdf(dx, (torch.arange(gw, device=self.dev) + 0.5) / gw) * (lx1 - lx0) - cx0) / (cx1 - cx0) * 2 - 1
        py, px_ = torch.meshgrid(cy, cx, indexing="ij")
        return z, torch.stack([px_, py], -1).float()

    @torch.inference_mode()
    def prepare(self, image, question):
        image = image.convert("RGB")
        w_img, h_img = image.size
        dev = self.dev
        thw = self.processor.image_processor(images=[image], return_tensors="pt")["image_grid_thw"][0]
        lr_img = image.resize((int(thw[2]) * 16, int(thw[1]) * 16), Image.BICUBIC)
        msgs = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [{"type": "image", "image": lr_img},
                                             {"type": "text", "text": question},
                                             {"type": "text", "text": dat_mod.WHERE_HINT}]}]
        proc = self.hr_processor if self.prefix_cache else self.processor
        text = proc.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False)
        text = text.rstrip()
        if text.endswith("<|im_end|>"):
            text = text[:-len("<|im_end|>")]
        inp = proc(text=[text], images=[lr_img], return_tensors="pt")
        inp = {k: v.to(dev) for k, v in inp.items() if torch.is_tensor(v)}
        inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
        out = self.model(**inp, output_hidden_states=True, return_dict=True,
                         use_cache=self.prefix_cache)
        prefix_ids = inp["input_ids"][0]
        prefix_cache = out.past_key_values if self.prefix_cache else None
        prefix_thw = inp["image_grid_thw"]
        ids = prefix_ids
        is_img = ids == self.image_id
        gh_l, gw_l = int(thw[1]) // 2, int(thw[2]) // 2
        hs = out.hidden_states[self.where_layer][0]
        cells = hs[is_img]
        if cells.shape[0] != gh_l * gw_l:
            print(f"[ldat] where-head grid {cells.shape[0]} != {gh_l}x{gw_l}", flush=True)
            return None
        logits = self.model.where_head(cells.reshape(gh_l, gw_l, -1), hs[int(torch.nonzero(is_img)[-1]) + 1:])
        head_p = torch.softmax(logits.float().flatten(), 0).reshape(logits.shape)
        hd_w, hd_h = dat_mod._where_hd_size(image, thw, 3, 5017600)
        side = self._mosaic_side()
        peaks = nms_peaks(logits, self.n_win, 4, 0.0)
        if not peaks:
            return None
        p = dat_mod._wdat_softmax(logits.float().tolist())
        wins, feats, pts = [], [], []
        for cy, cx in peaks:
            box = ((cx + 0.5 - 2) / gw_l, (cy + 0.5 - 2) / gh_l,
                   (cx + 0.5 + 2) / gw_l, (cy + 0.5 + 2) / gh_l)
            win = dat_mod._where_pad_window(box, 2.0, 20, hd_w, hd_h)
            tok_budget = side * side if side else self.crop_tok
            px0, py0, px1, py1, gh, gw = crop_geometry(image, win, tok_budget)
            if side:
                gh = gw = side
            lw = win
            if self.dat_k > 0:
                lw = dat_mod._wdat_adaptive_window(p, box, win, self.dat_k)[0]
            lx0, ly0 = int(lw[0] * w_img), int(lw[1] * h_img)
            lpx = (lx0, ly0, min(max(int(math.ceil(lw[2] * w_img)), lx0 + 1), w_img),
                   min(max(int(math.ceil(lw[3] * h_img)), ly0 + 1), h_img))
            lcrop = image.crop(lpx)
            lcw, lch = lcrop.size
            sc = math.sqrt(min(self.hr_cap * 1024, lcw * lch) / float(lcw * lch))
            nw, nh = max(32, int(lcw * sc) // 32 * 32), max(32, int(lch * sc) // 32 * 32)
            lmap = self._vit_image(lcrop.resize((nw, nh), Image.BICUBIC)).permute(2, 0, 1)[None].float()
            sw = min(lcw, round(self.src_scale * gw * 32))
            sh = min(lch, round(self.src_scale * gh * 32))
            psrc = torch.from_numpy(np.asarray(lcrop.resize((sw, sh), Image.BICUBIC), dtype=np.float32) / 255.0)
            psrc = psrc.permute(2, 0, 1).to(dev)
            win_rec = dict(px=(px0, py0, px1, py1), gh=gh, gw=gw, lpx=lpx,
                           lnbox=(lpx[0] / w_img, lpx[1] / h_img, lpx[2] / w_img, lpx[3] / h_img),
                           lmap=lmap, psrc=psrc, head_p=head_p)
            z, pt = self._ldat_tokens(win_rec)
            wins.append(win_rec)
            feats.append(z)
            pts.append(pt)

        dummies = [Image.new("RGB", (w["gw"] * 32, w["gh"] * 32)) for w in wins]
        qd = {"type": "text", "text": question}
        # Where-head still sees question + WHERE_HINT only. The answer turn is
        # the 2B layout: the question is asked again after the mosaic.
        content = [{"type": "image", "image": lr_img}, qd, {"type": "text", "text": dat_mod.WHERE_HINT}]
        content += [{"type": "image", "image": d} for d in dummies]
        content.append(dict(qd))
        amsgs = [{"role": "system", "content": "You are a helpful assistant."},
                 {"role": "user", "content": content}]
        atext = self.hr_processor.apply_chat_template(
            amsgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=self.enable_thinking)
        enc = self.hr_processor(text=[atext], images=[lr_img] + dummies, return_tensors="pt")
        thw2 = enc["image_grid_thw"].to(dev)
        for k, w in enumerate(wins):
            if tuple(thw2[1 + k].tolist()) != (1, 2 * w["gh"], 2 * w["gw"]):
                print(f"[ldat] mosaic {k} grid {tuple(thw2[1 + k].tolist())} != "
                      f"{(1, 2 * w['gh'], 2 * w['gw'])}", flush=True)
                return None
        ids = enc["input_ids"][0].to(dev)
        mm = torch.zeros_like(ids)
        mm[ids == self.image_id] = 1
        pos3d, rope_deltas = self.model.model.get_rope_index(ids[None], mm[None], image_grid_thw=thw2)
        pos3d = pos3d.to(dev)
        n_lr = int(torch.prod(thw2[0]))
        lr_pv = enc["pixel_values"][:n_lr]
        lr_thw = thw2[:1]
        img_pos = torch.nonzero(ids == self.image_id).flatten()
        n_win_tok = sum(w["gh"] * w["gw"] for w in wins)
        n_lr_tok = int(torch.prod(thw2[0]).item()) // (self.ms * self.ms)
        if img_pos.numel() != n_lr_tok + n_win_tok:
            print(f"[ldat] image tokens {img_pos.numel()} != {n_lr_tok}+{n_win_tok}", flush=True)
            return None
        runs = dat_mod._where_token_runs(ids, self.image_id)
        blocks = runs[1:]
        if len(blocks) != len(wins) or any(b - a != w["gh"] * w["gw"] for (a, b), w in zip(blocks, wins)):
            print("[ldat] window token runs do not match the mosaics", flush=True)
            return None
        emb = self.embed(ids[None])
        split = int(blocks[0][0])
        lr_end = int(runs[0][1])
        # The prefix stops at the hint. <|vision_start|> sits immediately before
        # the mosaic image tokens, so the cache is one token shorter than `split`
        # and that token is prefilled with the mosaic.
        npre = int(prefix_ids.shape[0]) if prefix_cache is not None else 0
        cache, cache_len = None, 0
        if (prefix_cache is not None and lr_end <= npre < int(ids.shape[0])
                and torch.equal(ids[:npre], prefix_ids)):
            same_grid = torch.equal(lr_thw, prefix_thw.to(device=lr_thw.device, dtype=lr_thw.dtype))
            if same_grid:
                pos_pre, _ = self.model.model.get_rope_index(
                    ids[:npre][None], mm[:npre][None], image_grid_thw=lr_thw)
                pos_pre = pos_pre.to(pos3d.device)
                if pos_pre.shape == pos3d[:, :, :npre].shape and torch.equal(
                        pos3d[:, :, :npre].long(), pos_pre.long()):
                    cache, cache_len = prefix_cache, npre
                else:
                    print("[ldat] prefix cache off: positions differ", flush=True)
            else:
                print("[ldat] prefix cache off: LR grid differs", flush=True)
        elif prefix_cache is not None:
            npre = int(prefix_ids.shape[0])
            ncmp = min(npre, split, int(ids.shape[0]))
            bad = -1
            if ncmp and not torch.equal(ids[:ncmp], prefix_ids[:ncmp]):
                bad = int((ids[:ncmp] != prefix_ids[:ncmp]).nonzero(as_tuple=False)[0])
            print(f"[ldat] prefix cache off: prefix {npre} tokens, mosaic starts at {split}, first mismatch {bad}",
                  flush=True)
        lr_emb = None
        if cache is None:
            lr_emb = self._vit(lr_pv.to(dev).to(torch.bfloat16), lr_thw)
        return dict(ids=ids, mm=mm[None], pos3d=pos3d, rope_deltas=rope_deltas, emb=emb,
                    lr_emb=lr_emb, lr_pv=lr_pv, lr_thw=lr_thw, img_pos=img_pos, blocks=blocks,
                    wins=wins, feats=feats, pts=pts, n_tokens=int(ids.shape[0]),
                    cache=cache, cache_len=cache_len, split=split)

    def _lr_emb(self, pr):
        if pr["lr_emb"] is None:
            pr["lr_emb"] = self._vit(pr["lr_pv"].to(self.dev).to(torch.bfloat16), pr["lr_thw"])
        return pr["lr_emb"]

    def _prefill(self, pr):
        dev = self.dev
        pos = global_positions(pr["pos3d"], pr["blocks"], pr["wins"], pr["pts"])
        self.model.model.rope_deltas = pr["rope_deltas"]
        n = pr["emb"].shape[1]
        cut = pr["cache_len"]
        if pr["cache"] is not None:
            e = pr["emb"][:, cut:].clone()
            rel = pr["img_pos"]
            rel = rel[rel >= cut] - cut
            e[0, rel] = torch.cat(pr["feats"], 0).to(e.dtype)
            o = self.model(inputs_embeds=e, position_ids=pos[:, :, cut:],
                           mm_token_type_ids=pr["mm"][:, cut:],
                           past_key_values=pr["cache"],
                           cache_position=torch.arange(cut, n, device=dev),
                           use_cache=True, return_dict=True, logits_to_keep=1)
            self.n_cache += 1
            if self.n_cache == 1:
                gap = self.tok.decode(pr["ids"][cut:pr["split"]].tolist())
                print(f"[ldat] prefix cache reused {cut} tokens, then {gap!r}, new {n - cut}",
                      flush=True)
        else:
            e = pr["emb"].clone()
            e[0, pr["img_pos"]] = torch.cat([self._lr_emb(pr)] + pr["feats"], 0).to(e.dtype)
            o = self.model(inputs_embeds=e, position_ids=pos, mm_token_type_ids=pr["mm"],
                           cache_position=torch.arange(n, device=dev),
                           use_cache=True, return_dict=True, logits_to_keep=1)
        return o, pos, n

    @torch.inference_mode()
    def generate(self, image, question, max_new_tokens):
        pr = self.prepare(image, question)
        if pr is None:
            return ""
        self.last_n_tokens = int(pr["n_tokens"])
        dev = self.dev
        try:
            o, pos, n = self._prefill(pr)
        except Exception as exc:
            self.n_cache_fallback += 1
            print(f"[ldat] prefix cache failed ({type(exc).__name__}: {exc}); full prefill", flush=True)
            pr["cache"] = None
            o, pos, n = self._prefill(pr)
        cache = o.past_key_values
        next_pos = int(math.floor(float(pos.max()))) + 1
        logits = o.logits[0, -1]
        tok = int(logits.argmax())
        toks = []
        # A finished think is </think> and then a short answer. The 1024 cap was
        # cutting 42/64 unfinished traces mid-sentence, and <|im_end|> was ending
        # the other 22 before the tag. Keep going until the tag, then 48 answer
        # tokens. An end token in the first 16 tokens still stops, so a bare
        # letter is unchanged.
        think_limit = max(int(max_new_tokens), 4096 if self.think_end is not None else 1)
        answer_limit = 48
        closed = False
        answer_left = answer_limit
        while True:
            if tok in self.eos and (self.think_end is None or closed or len(toks) < 16):
                break
            if tok in self.eos:
                masked = logits.float().clone()
                for eid in self.eos:
                    if 0 <= eid < masked.shape[-1]:
                        masked[eid] = torch.finfo(masked.dtype).min
                tok = int(masked.argmax())
                if tok in self.eos:
                    break
            if self.think_end is not None and not closed and len(toks) >= think_limit:
                break
            if self.think_end is None and len(toks) >= think_limit:
                break
            toks.append(tok)
            just_closed = self.think_end is not None and tok == self.think_end and not closed
            if just_closed:
                closed = True
                answer_left = answer_limit
            elif closed:
                answer_left -= 1
                if answer_left <= 0:
                    break
            k = len(toks) - 1
            o = self.model(input_ids=torch.tensor([[tok]], device=dev),
                           position_ids=torch.full((3, 1, 1), next_pos + k, device=dev, dtype=torch.long),
                           past_key_values=cache,
                           cache_position=torch.tensor([n + k], device=dev),
                           use_cache=True, return_dict=True)
            cache = o.past_key_values
            logits = o.logits[0, -1]
            tok = int(logits.argmax())
        if self.think_end is not None:
            if closed:
                self.n_think_closed += 1
            else:
                self.n_think_capped += 1
        # Thinking mode emits a </think> token. skip_special_tokens drops it, and
        # the letter extractor then sees the reasoning instead of the answer.
        text = self.tok.decode(toks, skip_special_tokens=not self.enable_thinking)
        if self.enable_thinking:
            for tag in ("<|im_end|>", "<|im_start|>", "<|endoftext|>"):
                text = text.replace(tag, "")
        return text.strip()
