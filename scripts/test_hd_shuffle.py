#!/usr/bin/env python3
"""CPU regressions for the probes' wrong-image pairing and HD geometry.

Run with: python scripts/test_hd_shuffle.py
Requires Pillow, but no model, checkpoint, CUDA, or PyTorch.
"""

import ast
import os
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.hd_probe_utils import different_image_indices, image_fingerprint


def load_probe_functions(filename, names):
    """Execute production helpers without importing the model/CUDA dependencies."""
    path = ROOT / "scripts" / filename
    tree = ast.parse(path.read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in functions} == set(names)
    namespace = {"Image": Image, "os": os,
                 "different_image_indices": different_image_indices,
                 "image_fingerprint": image_fingerprint}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class HDShuffleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.probe = load_probe_functions(
            "probe_whd_qwen35.py", {"prepare_shuffle_sources", "make_hd_image"})
        cls.leverage = load_probe_functions("_test_lr_drop_leverage.py", {"prepare_shuffle_pairs"})

    def test_full_image_hash_distinguishes_shared_blank_header(self):
        first = Image.new("RGB", (64, 64), "white")
        second = first.copy()
        second.putpixel((63, 63), (0, 0, 0))
        self.assertEqual(first.tobytes()[:4096], second.tobytes()[:4096])
        self.assertNotEqual(image_fingerprint(first), image_fingerprint(second))
        samples = [{"image": first}, {"image": second}]
        self.probe["prepare_shuffle_sources"](samples)
        self.assertEqual([s["shuffle_source_index"] for s in samples], [1, 0])

    def test_search_reaches_donor_beyond_old_eight_candidate_limit(self):
        ids = ["same"] * 24 + ["different"]
        donors = different_image_indices(ids)
        self.assertEqual(donors[0], 24)
        self.assertTrue(all(ids[i] != ids[j] for i, j in enumerate(donors)))
        self.assertEqual(donors, different_image_indices(ids))

    def test_repeated_images_never_become_each_others_donors(self):
        # Half-way pairing would map the first A to another A.
        ids = ["A", "B", "A", "B"]
        donors = different_image_indices(ids)
        self.assertTrue(all(ids[i] != ids[j] for i, j in enumerate(donors)))

    def test_single_content_pool_fails_explicitly(self):
        for ids in ([], ["A"], ["A"] * 20):
            with self.subTest(ids=ids):
                with self.assertRaisesRegex(ValueError, "at least two different image contents"):
                    different_image_indices(ids)

    def test_fingerprint_normalizes_rgb_and_includes_dimensions(self):
        image = Image.new("L", (10, 20), 32)
        self.assertEqual(image_fingerprint(image), image_fingerprint(image.convert("RGB")))
        self.assertNotEqual(image_fingerprint(image), image_fingerprint(Image.new("L", (20, 10), 32)))

    def test_hd_replacement_keeps_target_geometry_and_original_image(self):
        original = Image.new("RGB", (32, 16), "red")
        donor = Image.new("RGB", (8, 64), "blue")
        samples = [{"image": original, "prompt": "original question"}, {"image": donor}]
        self.probe["prepare_shuffle_sources"](samples)
        output = self.probe["make_hd_image"](
            samples[0], 0, samples, None, 96, 48, "shuffle")
        self.assertEqual(output.size, (96, 48))
        self.assertEqual(output.getpixel((0, 0)), (0, 0, 255))
        self.assertEqual(original.size, (32, 16))
        self.assertEqual(original.getpixel((0, 0)), (255, 0, 0))
        self.assertEqual(samples[0]["prompt"], "original question")
        self.assertEqual(samples[0]["shuffle_source_sha256"], image_fingerprint(donor))
        self.assertNotEqual(samples[0]["image_sha256"], samples[0]["shuffle_source_sha256"])

    def test_non_shuffle_source_still_accepts_single_image(self):
        sample = {"image": Image.new("RGB", (5, 7), "red")}
        output = self.probe["make_hd_image"](sample, 0, [sample], None, 8, 10, "real")
        self.assertEqual(output.size, (8, 10))
        self.assertNotIn("shuffle_source_index", sample)

    def test_leverage_pairing_excludes_duplicate_files_and_records_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / f"{i}.png" for i in range(3)]
            Image.new("RGB", (12, 12), "red").save(paths[0])
            Image.new("RGB", (12, 12), "red").save(paths[1])
            Image.new("RGB", (7, 19), "blue").save(paths[2])
            order = [paths[0], paths[2], paths[1], paths[2]]
            items = [(str(path), "question", "answer", [0, 0, 1, 1]) for path in order]
            pairs = self.leverage["prepare_shuffle_pairs"](items)
            self.assertEqual(len(pairs), len(items))
            for i, pair in enumerate(pairs):
                self.assertEqual(pair["sample_index"], i)
                self.assertEqual(pair["image"], str(order[i]))
                self.assertEqual(pair["shuffle_source_image"], str(order[pair["shuffle_source_index"]]))
                self.assertNotEqual(pair["image_sha256"], pair["shuffle_source_sha256"])
            self.assertEqual(pairs[0]["image_sha256"], pairs[2]["image_sha256"])


if __name__ == "__main__":
    unittest.main()
