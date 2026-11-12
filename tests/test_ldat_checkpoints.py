import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from ldat_eval.checkpoints import load_head, save_head


class CheckpointTests(unittest.TestCase):
    def test_legacy_tensor_format_and_wrong_backbone_rejection(self):
        b = SimpleNamespace(family="qwen2_5_vl", hidden_size=64, factor=28)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "where.pt"
            state = {"img_in.0.weight": torch.ones(64)}
            save_head(path, state, metadata=dict(kind="where", model_family="qwen2_5_vl",
                                                hidden_size=64, factor=28, where_layer=1))
            # Existing standalone probes can still consume the raw state dict.
            self.assertEqual(set(torch.load(path, weights_only=True)), set(state))
            got, _ = load_head(path, b, kind="where", where_layer=1)
            torch.testing.assert_close(got["img_in.0.weight"], state["img_in.0.weight"])
            with self.assertRaisesRegex(ValueError, "where_layer"):
                load_head(path, b, kind="where", where_layer=8)
            with self.assertRaisesRegex(ValueError, "model_family"):
                load_head(path, SimpleNamespace(family="qwen3_5", hidden_size=64, factor=32), kind="where")

    def test_unmerged_lora_cannot_silently_disappear_in_evaluation(self):
        b = SimpleNamespace(family="qwen2_5_vl", hidden_size=64, factor=28)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ldat.pt"
            torch.save({"head": {"inp.0.weight": torch.ones(64)}, "lora": {"q": torch.ones(1)}}, path)
            with self.assertRaisesRegex(ValueError, "LoRA/window-head"):
                load_head(path, b, kind="ldat", inference=True)


if __name__ == "__main__":
    unittest.main()
