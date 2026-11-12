"""CPU integration checks for the learned sampler's actual HF backbone boundary."""
from types import SimpleNamespace
import unittest

import torch
from PIL import Image
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration


def tiny_qwen25():
    config = Qwen2_5_VLConfig(
        text_config=dict(vocab_size=128, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                                          "mrope_section": [2, 3, 3]}),
        vision_config=dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=4,
                           out_hidden_size=64, patch_size=14, spatial_merge_size=2,
                           temporal_patch_size=2, window_size=112, fullatt_block_indexes=[0, 1]),
        image_token_id=120, video_token_id=121, vision_start_token_id=118,
        vision_end_token_id=119,
    )
    config._attn_implementation = "sdpa"
    return Qwen2_5_VLForConditionalGeneration(config).eval()


class BackboneTests(unittest.TestCase):
    def setUp(self):
        from ldat_eval.backbone import VLMBackbone
        torch.manual_seed(7)
        self.model = tiny_qwen25()
        self.backbone = VLMBackbone(self.model, "qwen2_5_vl")

    def test_visual_tokens_and_pixel_gradient(self):
        # One 56x84 image -> 4x6 patches -> 2x3=6 merged tokens.
        pv = torch.randn(24, 3 * 2 * 14 * 14, requires_grad=True)
        for p in self.model.parameters():
            p.requires_grad_(False)
        z = self.backbone.visual_tokens(pv, torch.tensor([[1, 4, 6]]))
        self.assertEqual(tuple(z.shape), (6, 64))
        z.square().sum().backward()
        self.assertTrue(torch.isfinite(pv.grad).all())
        self.assertGreater(float(pv.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_rope_and_cached_fractional_position_continuation(self):
        ids = torch.tensor([[7, 118, 120, 120, 120, 120, 120, 120, 119, 8]])
        mm = (ids == 120).long()
        pos, delta = self.backbone.rope_index(ids, mm, torch.tensor([[1, 4, 6]]))
        self.assertEqual(tuple(pos.shape), (3, 1, 10))
        torch.testing.assert_close(pos[1, 0, 2:8], torch.tensor([2, 2, 2, 3, 3, 3]))
        torch.testing.assert_close(pos[2, 0, 2:8], torch.tensor([2, 3, 4, 2, 3, 4]))
        pos = pos.float()
        pos[1:, :, 2:8] += 0.25
        emb = self.model.get_input_embeddings()(ids)
        out = self.backbone.forward(inputs_embeds=emb, position_ids=pos,
                                    mm_token_type_ids=mm, use_cache=True, return_dict=True)
        tail_id = torch.tensor([[9]])
        tail_pos = torch.full((3, 1, 1), int(pos.max()) + 1)
        tail = self.backbone.forward(input_ids=tail_id, position_ids=tail_pos,
                                     past_key_values=out.past_key_values, use_cache=True,
                                     cache_position=torch.tensor([10]), return_dict=True)
        full = self.backbone.forward(input_ids=torch.cat([ids, tail_id], 1),
                                     position_ids=torch.cat([pos, tail_pos], -1),
                                     use_cache=False, return_dict=True)
        torch.testing.assert_close(tail.logits[:, -1], full.logits[:, -1], atol=2e-5, rtol=2e-5)

    def test_geometry_uses_model_token_units(self):
        self.assertEqual((self.backbone.patch_size, self.backbone.factor, self.backbone.hidden_size),
                         (14, 28, 64))
        from llava.model.where_dat import _where_hd_size, _where_pad_window
        from scripts.patch_sample_probe import crop_geometry
        img = Image.new("RGB", (3360, 2240))
        self.assertEqual(_where_hd_size(img, [1, 16, 24], 3, 5017600,
                                       patch_size=14, merge_size=2), (1008, 672))
        box = _where_pad_window((0.45, 0.45, 0.55, 0.55), 1, 4, 280, 280, factor=28)
        self.assertAlmostEqual(box[2] - box[0], 0.4)
        self.assertEqual(crop_geometry(Image.new("RGB", (280, 140)), (0, 0, 1, 1), 50,
                                       factor=28), (0, 0, 280, 140, 5, 10))

    def test_analytic_dat_control_quantiles_and_default_geometry(self):
        from llava.model.where_dat import _wdat_inv_cdf_edges, _where_dat_image
        self.assertEqual(_wdat_inv_cdf_edges([1.0, 1.0], 4), [0.0, 0.25, 0.5, 0.75, 1.0])
        img = Image.new("RGB", (320, 160))
        for factor, size in ((32, (320, 160)), (28, (280, 140))):
            sampled, _ = _where_dat_image(img, (0, 0, 1, 1), 50, torch.zeros(5, 10),
                                          (0.25, 0.25, 0.75, 0.75), k=None, factor=factor)
            self.assertEqual(sampled.size, size)


if __name__ == "__main__":
    unittest.main()
