#!/usr/bin/env python3
"""Exercise the production probe report with NumPy/Pillow, without Torch/GPU.

Run from the repository root: python scripts/test_probe_report.py
The AST extraction skips model loading and inference, but executes the actual
report statements through the final JSON write rather than a copied reporter.
"""

import ast
import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image


class ProbeReportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).with_name("probe_whd_qwen35.py")
        tree = ast.parse(source.read_text(), filename=str(source))
        main = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "main")
        start = next(i for i, node in enumerate(main.body)
                     if isinstance(node, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == "cats"
                             for t in node.targets))
        module = ast.parse("def run_report():\n    pass\n")
        module.body[0].body = main.body[start:]
        cls.report_code = compile(ast.fix_missing_locations(module), str(source), "exec")

    def fixture(self, count=6, constant=False, rich=False):
        samples = [{"category": "bbox", "gt": "a", "prompt": "Which letter?",
                    "image": Image.new("RGB", (100, 100)),
                    "bboxes": [[10, 10, 30, 1 if i % 2 else 30]]}
                   for i in range(count)]
        hd_rows = []
        global_rows = []
        for i in range(count):
            x, y = (0.1, 0.2) if constant else (0.05 * i, 0.1 * i)
            hd_rows.append((x, y, x, y, 1.0, 0.4, 0.1, x, y,
                            [-0.4, -0.4, 0.4, 0.4], 0.6, 0.3))
            global_rows.append((x, y, 0.6, 0.6, x, y, 0.3, 0.3))
        configs = ["off", "real b=+0"]
        preds = {"off": ["a" if i % 2 == 0 else "b" for i in range(count)],
                 "real b=+0": ["a"] * count}
        namespace = {
            "np": np, "math": math, "json": json, "Image": Image,
            "samples": samples, "configs": configs, "preds": preds,
            "raw": preds, "hit": lambda p, g: bool(g) and g in p,
            "has_gt": True, "free_text": True, "record_bias": 0.0,
            "TOK_PX": 1024, "WHD": {}, "KHD": {}, "KLR": {},
            "LOC": {}, "GLOB": {}, "GLOBC": {}, "GLOBP": {},
            "HDC": {11: hd_rows},
            "args": SimpleNamespace(tok_budget=256, hd_source="real", grid_size=20,
                                    model_path="fixture", image_folder=None,
                                    dataset="synth", hd_bias=[0.0], dump_maps=None),
        }
        if constant:
            namespace["GLOBC"][11] = global_rows
        if rich:
            for layer in (11, 15):
                namespace["GLOB"][layer] = [(0.1, 0.6)] * count
                namespace["GLOBC"][layer] = list(global_rows)
                namespace["HDC"][layer] = list(hd_rows)
                namespace["LOC"][layer] = [(0.1, 0.2, 0.3, 0.1)] * count
                namespace["WHD"][layer] = [(0.1, 0.2, 0.3)] * count
                namespace["KHD"][layer] = [(0.5, 0.1)] * count
                namespace["KLR"][layer] = [0.5] * count
                namespace["GLOBP"][layer] = []
                for i in range(count):
                    probability = np.full(400, 0.5 / 399)
                    probability[i if i % 2 == 0 else 200 + i] = 0.5
                    window = np.zeros(400, dtype=bool)
                    window[:100] = True
                    heads = np.tile(probability, (8, 1))
                    alternatives = {key: heads for key in (
                        "q_last", "im_end", "nl", "asst", "asst_nl", "q_mean")}
                    namespace["GLOBP"][layer].append(
                        (probability, window, window, (20, 20), probability,
                         heads, alternatives))
        return namespace

    def run_report(self, namespace):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "report.json"
            namespace["args"].out = str(output)
            exec(self.report_code, namespace)
            printed = io.StringIO()
            with contextlib.redirect_stdout(printed):
                namespace["run_report"]()
            self.assertTrue(output.is_file())
            return json.loads(output.read_text()), printed.getvalue()

    def test_hd_diagnostics_without_global_offset(self):
        result, printed = self.run_report(self.fixture())
        stats = result["layer_glob"]["11"]
        self.assertEqual(stats["hdc_all"]["n"], 6)
        self.assertEqual(stats["hdc_covered"]["n"], 6)
        self.assertAlmostEqual(stats["hdc_all"]["r_cx"], 1.0)
        self.assertTrue(math.isnan(stats["hdc_all"]["d_lr"]))
        self.assertEqual(result["num_samples"], 6)
        self.assertIn("HD-key refinement (diag 12)", printed)

    def test_small_hd_sample_skips_correlation(self):
        result, _ = self.run_report(self.fixture(count=4))
        self.assertEqual(result["num_samples"], 4)
        self.assertEqual(result["layer_glob"], {})

    def test_constant_coordinates_have_undefined_correlation(self):
        result, _ = self.run_report(self.fixture(constant=True))
        stats = result["layer_glob"]["11"]
        self.assertTrue(math.isnan(stats["r_cx"]))
        self.assertTrue(math.isnan(stats["hdc_all"]["r_cx"]))
        self.assertTrue(math.isnan(stats["hdc_covered"]["r_cy"]))

    def test_full_report_keeps_accuracy_and_serializes(self):
        result, printed = self.run_report(self.fixture(rich=True))
        self.assertEqual(result["num_samples"], 6)
        self.assertEqual(len(result["per_sample"]), 6)
        self.assertEqual(result["summary"]["off"]["acc"], 50.0)
        self.assertEqual(result["summary"]["real b=+0"]["acc"], 100.0)
        self.assertEqual(result["summary"]["real b=+0"]["fixed"], 3)
        self.assertEqual(result["summary"]["real b=+0"]["broke"], 0)
        combined = result["layer_glob"]["-1"]
        self.assertIn("fusion", combined)
        self.assertIn("head_pool", combined)
        self.assertEqual(combined["by_outcome"]["peak in win"]["n"], 3)
        self.assertEqual(combined["by_outcome"]["peak out"]["n"], 3)
        self.assertIn("head pooling", printed)


if __name__ == "__main__":
    unittest.main()
