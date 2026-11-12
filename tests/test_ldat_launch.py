import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from test_ldat_pipeline import save_tiny_model

ROOT = Path(__file__).resolve().parents[1]


class LaunchTests(unittest.TestCase):
    def test_native_lora_merge_preserves_trained_merger(self):
        from peft import LoraConfig, get_peft_model
        from transformers import Qwen2_5_VLForConditionalGeneration
        from scripts.merge_native_lora import merge
        with tempfile.TemporaryDirectory() as d:
            base, adapter, out = (Path(d) / x for x in ("base", "adapter", "merged"))
            model, _ = save_tiny_model(base)
            model = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"]))
            for name, p in model.named_parameters():
                if "lora_B" in name:
                    with torch.no_grad():
                        p.fill_(0.01)
            name, p = next((n, p) for n, p in model.named_parameters() if "visual.merger.mlp.0.weight" in n)
            with torch.no_grad():
                p.add_(0.1)
            expected_merger = p.detach().clone()
            model.save_pretrained(adapter)
            torch.save({name: expected_merger}, adapter / "non_lora_trainables.bin")
            ids = torch.tensor([[4, 7, 8]])
            with torch.no_grad():
                expected = model(input_ids=ids, use_cache=False).logits
            merge(str(base), str(adapter), str(out), dtype=torch.float32)
            merged = Qwen2_5_VLForConditionalGeneration.from_pretrained(out).eval()
            torch.testing.assert_close(merged.model.visual.merger.mlp[0].weight, expected_merger)
            with torch.no_grad():
                torch.testing.assert_close(merged(input_ids=ids, use_cache=False).logits,
                                           expected, atol=2e-5, rtol=2e-5)

    def test_dry_run_trains_and_evaluates_same_backbone_and_budget(self):
        with tempfile.TemporaryDirectory() as d:
            env = dict(os.environ, DRY_RUN="1", OUT=str(Path(d) / "run"), RUN_ROOT=str(Path(d) / "run"),
                       MODEL_PATH="/model/Qwen2.5-VL-7B-Instruct", STAGES="where dat",
                       LR_TOK="320", CROP_TOK="200", TASKS="vstar_bench", NPROC="2")
            for name in ("train_7b.sh", "eval_7b.sh"):
                res = subprocess.run(["bash", str(ROOT / "scripts/qwen2_5_ldat" / name)],
                                     env=env, capture_output=True, text=True)
                self.assertEqual(res.returncode, 0, res.stderr)
                self.assertIn("Qwen2.5-VL-7B-Instruct", res.stdout)
                self.assertIn("320", res.stdout)
                self.assertIn("200", res.stdout)
            self.assertFalse((Path(d) / "run").exists())

    def test_gpu_smoke_command_on_cpu_tiny_model(self):
        with tempfile.TemporaryDirectory() as d:
            save_tiny_model(Path(d))
            res = subprocess.run([sys.executable, "scripts/qwen2_5_ldat/smoke.py", "--model_path", d,
                                  "--device", "cpu", "--where_layer", "1", "--lr_tok", "6",
                                  "--crop_tok", "4", "--hr_cap", "12", "--backward"],
                                 cwd=ROOT, capture_output=True, text=True, timeout=90)
            self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
            self.assertIn("[PASS]", res.stdout)


if __name__ == "__main__":
    unittest.main()
