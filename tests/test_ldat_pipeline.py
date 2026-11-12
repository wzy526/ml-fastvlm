"""Offline, real-model tests: pixels -> learned density -> ViT -> LM -> cache."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from test_ldat_backbone import tiny_qwen25


def save_tiny_model(path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2_5_VLProcessor, Qwen2VLImageProcessor
    from transformers.models.qwen2_vl.video_processing_qwen2_vl import Qwen2VLVideoProcessor
    vocab = {f"unused{i}": i for i in range(128)}
    for i, word in {0: "[UNK]", 1: "[PAD]", 2: "<|im_start|>", 3: "<|im_end|>",
                    4: "user", 5: "assistant", 6: "system", 7: "What", 8: "color", 9: "red",
                    118: "<|vision_start|>", 119: "<|vision_end|>",
                    120: "<|image_pad|>", 121: "<|video_pad|>"}.items():
        del vocab[f"unused{i}"]
        vocab[word] = i
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]",
                                 eos_token="<|im_end|>", additional_special_tokens=[
                                     "<|im_start|>", "<|vision_start|>", "<|vision_end|>",
                                     "<|image_pad|>", "<|video_pad|>"])
    template = """{% for m in messages %}<|im_start|>{{ m['role'] }}
{% if m['content'] is string %}{{ m['content'] }}{% else %}{% for c in m['content'] %}{% if c['type'] == 'image' %}<|vision_start|><|image_pad|><|vision_end|>{% else %} {{ c['text'] }} {% endif %}{% endfor %}{% endif %}<|im_end|>
{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}"""
    proc = Qwen2_5_VLProcessor(image_processor=Qwen2VLImageProcessor(patch_size=14, merge_size=2,
                                  temporal_patch_size=2, min_pixels=784, max_pixels=4704),
                               tokenizer=tok, chat_template=template, video_processor=Qwen2VLVideoProcessor())
    model = tiny_qwen25()
    model.generation_config.eos_token_id = 3
    model.save_pretrained(path)
    proc.save_pretrained(path)
    return model, proc


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.model, cls.proc = save_tiny_model(cls.root)
        from llava.model.where_dat import _WhereXAttnHead
        from ldat_eval.infer import LDatHead
        torch.manual_seed(11)
        torch.save(_WhereXAttnHead(64, 64).state_dict(), cls.root / "where.pt")
        torch.save(LDatHead(64, d=16).state_dict(), cls.root / "ldat.pt")
        cls.img = Image.fromarray(np.random.default_rng(9).integers(0, 255, (112, 168, 3), dtype=np.uint8))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def runner(self, **kw):
        from ldat_eval.infer import LDatRunner
        defaults = dict(model_family="qwen2_5_vl", where_layer=1, lr_tok=6,
                        crop_tok=4, hr_cap=12, ldat_d=16, supersample=1)
        defaults.update(kw)
        return LDatRunner(str(self.root), str(self.root / "where.pt"), str(self.root / "ldat.pt"),
                          "cpu", **defaults)

    def test_full_fast_and_cached_prefills_agree(self):
        runner = self.runner()
        pr = runner.prepare(self.img, "What color")
        self.assertIsNotNone(pr)
        full, pos, n = runner._prefill(pr)
        expected = full.logits.detach().clone()
        self.assertGreater(n, 6)
        self.assertEqual(runner.gdn_multitoken, "not_applicable")
        runner.fast_prefix = True
        fast = runner.prepare(self.img, "What color")
        fast_out, _, _ = runner._prefill(fast)
        torch.testing.assert_close(fast_out.logits, expected, atol=2e-5, rtol=2e-5)
        self.assertEqual(len(runner.backbone.language_model.layers), 2)
        runner.prefix_cache = True
        cached = runner.prepare(self.img, "What color")
        self.assertIsNotNone(cached["cache"])
        cache_out, _, _ = runner._prefill(cached)
        torch.testing.assert_close(cache_out.logits, expected, atol=2e-5, rtol=2e-5)
        result = runner.generate(self.img, "What color", max_new_tokens=2)
        self.assertIsInstance(result, str)

    def test_sampler_gradient_through_frozen_vit_and_lm(self):
        runner = self.runner()
        pr = runner.prepare(self.img, "What color")
        # prepare() is inference-only. Training tensors must be ordinary tensors.
        win = {k: v.clone() if torch.is_tensor(v) else v for k, v in pr["wins"][0].items()}
        z, _ = runner._ldat_tokens(win)
        self.assertTrue(z.requires_grad)
        pos = torch.arange(z.shape[0] + 1).view(1, 1, -1).expand(3, 1, -1)
        e = torch.cat([z[None], runner.embed(torch.tensor([[9]]))], 1)
        out = runner.backbone.forward(inputs_embeds=e, position_ids=pos, use_cache=False)
        torch.nn.functional.cross_entropy(out.logits[:, -2], torch.tensor([9])).backward()
        self.assertGreater(float(runner.head.out.weight.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None for p in runner.model.parameters()))

    def test_two_windows_preserve_exact_token_counts(self):
        runner = self.runner(n_win=2, grid=2, lr_tok=96)
        pr = runner.prepare(self.img.resize((336, 224)), "What color")
        self.assertEqual([b - a for a, b in pr["blocks"]], [4, 4])
        self.assertEqual([z.shape[0] for z in pr["feats"]], [4, 4])
        out, _, _ = runner._prefill(pr)
        self.assertTrue(torch.isfinite(out.logits).all())

    def test_training_entrypoints_save_reloadable_heads(self):
        image_path = self.root / "image.png"
        self.img.save(image_path)
        data = self.root / "data.json"
        data.write_text(json.dumps([dict(image=str(image_path), bbox=[0.2, 0.2, 0.8, 0.8],
            conversations=[dict(**{"from": "human", "value": "<image>\nWhat color"}),
                           dict(**{"from": "gpt", "value": "red"})]) for _ in range(3)]))
        where = self.root / "trained_where.pt"
        learned = self.root / "trained_ldat.pt"
        root = Path(__file__).resolve().parents[1]
        common = ["--model_path", str(self.root), "--model_family", "qwen2_5_vl", "--device", "cpu"]
        commands = [
            ["scripts/where_head.py", *common, "--train_json", str(data), "--head", "xattn",
             "--d", "384", "--feat", "lm:1", "--head_prefix", "train", "--lr_tok", "6",
             "--vstar_n", "0", "--heldout", "0", "--steps", "1", "--accum", "1",
             "--stagger", "0", "--save", str(where)],
            ["scripts/offset_feat_train.py", *common, "--where_head_pt", str(where),
             "--where_layer", "1", "--train_json", str(data), "--held_from_train", "1",
             "--max_train", "2", "--steps", "1", "--accum", "1", "--cfgs", "crop,dat,ldat",
             "--train_cfg", "ldat", "--ldat_in", "native", "--ldat_d", "16", "--lr_tok", "6",
             "--n_win", "1", "--crop_tok", "4", "--hr_cap", "12", "--supersample", "1",
             "--save", str(learned)],
        ]
        for cmd in commands:
            result = subprocess.run([sys.executable, *cmd], cwd=root, capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        from ldat_eval.infer import LDatRunner
        runner = LDatRunner(str(self.root), str(where), str(learned), "cpu", model_family="qwen2_5_vl",
                           where_layer=1, ldat_d=16, lr_tok=6, crop_tok=4, hr_cap=12, supersample=1)
        self.assertGreater(float(runner.head.out.weight.detach().abs().sum()), 0)
        self.assertIsInstance(runner.generate(self.img, "What color", 2), str)

    def test_where_training_sees_identical_prefix_pixels_and_states(self):
        from where_head import Frozen
        from ldat_eval.backbone import where_prefix
        frozen = Frozen(str(self.root), 1, 5017600, "sdpa", "cpu",
                        model_family="qwen2_5_vl", head_prefix="train")
        frozen.set_lm_view(str(self.root), 1, 6)
        cells, txt = frozen.pair(self.img, "What color")
        runner = self.runner(fast_prefix=True)
        thw = runner.processor.image_processor(images=[self.img], return_tensors="pt")["image_grid_thw"][0]
        lr = self.img.resize((int(thw[2]) * 14, int(thw[1]) * 14), Image.BICUBIC)
        inp = runner.processor(text=[where_prefix(runner.processor, lr, "What color")],
                               images=[lr], return_tensors="pt")
        with torch.no_grad():
            hs, _ = runner._where_states(inp)
        mask = inp["input_ids"][0] == 120
        torch.testing.assert_close(cells.flatten(0, 1), hs[mask], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(txt, hs[int(mask.nonzero()[-1]) + 1:], atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
