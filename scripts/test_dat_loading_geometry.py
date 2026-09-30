#!/usr/bin/env python3
"""CPU regressions for original fp32 DAT reloads and LR-first pixel units."""
import ast
import copy
import logging
import math
import os
from pathlib import Path
import tempfile
import unittest

try:
    import torch
    from safetensors.torch import save_file
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'llava/model/language_model/modeling_qwen3_5_dat.py'
TRAIN = ROOT / 'llava/train/train_qwen_dat.py'


def extract_class(path, class_name, methods, namespace, base='object', attributes=()):
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.ClassDef) and n.name == class_name)
    body = [copy.deepcopy(n) for n in node.body
            if isinstance(n, ast.FunctionDef) and n.name in methods
            or isinstance(n, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in attributes for t in n.targets)]
    cls = ast.ClassDef(name=class_name, bases=[ast.Name(id=base, ctx=ast.Load())],
                       keywords=[], body=body, decorator_list=[])
    module = ast.Module(body=[ast.ImportFrom(module='__future__',
        names=[ast.alias(name='annotations')], level=0), cls], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    return namespace[class_name]


@unittest.skipUnless(torch is not None, 'requires torch and safetensors')
class LoadingTests(unittest.TestCase):
    def test_auto_and_bf16_preserve_all_66_original_tensors(self):
        class Base:
            _keep_in_fp32_modules = ['base_norm']

            @classmethod
            def from_pretrained(cls, path, **kwargs):
                # Emulate an HF version that casts everything to its load dtype.
                model = torch.nn.Module()
                model.register_parameter('reader', torch.nn.Parameter(torch.zeros(4, dtype=torch.bfloat16)))
                model.targets = {k: torch.nn.Parameter(v.to(torch.bfloat16)) for k, v in source.items()}
                model.named_parameters = lambda: model.targets.items()
                return model

        ns = dict(torch=torch, os=os, logger=logging.getLogger(__name__),
                  Qwen3_5ForConditionalGeneration=Base)
        cls = extract_class(MODEL, 'Qwen3_5DATForConditionalGeneration',
            {'from_pretrained', '_manual_load_dat_raw_params'}, ns,
            base='Qwen3_5ForConditionalGeneration', attributes={
                '_DAT_FP32_MODULES', '_keep_in_fp32_modules', '_DAT_REINIT_MARKERS'})
        source = {}
        for layer in range(6):
            for component in cls._DAT_FP32_MODULES:
                suffix = '' if component == 'hd_gate' else '.weight'
                source[f'model.language_model.layers.{layer}.self_attn.{component}{suffix}'] = (
                    torch.arange(4).float() * 0.000137 + 1.000123)
        self.assertEqual(len(source), 66)
        reader_key = 'model.language_model.layers.0.self_attn.k_proj_hd.weight'
        source[reader_key] = torch.ones(4)
        with tempfile.TemporaryDirectory() as directory:
            save_file(source, str(Path(directory) / 'model.safetensors'))
            for dtype in ('auto', torch.bfloat16):
                model = cls.from_pretrained(Path(directory), torch_dtype=dtype)
                for key, value in model.targets.items():
                    if key == reader_key:
                        self.assertEqual(value.dtype, torch.bfloat16)
                    else:
                        self.assertEqual(value.dtype, torch.float32)
                        self.assertTrue(torch.equal(value, source[key]), key)
        self.assertIn('base_norm', cls._keep_in_fp32_modules)


class GeometryAndSeedTests(unittest.TestCase):
    def test_probe_and_bench_use_patch_units_and_merged_alignment(self):
        for script in ('probe_whd_qwen35.py', 'bench_ttft_qwen3_5.py'):
            tree = ast.parse((ROOT / 'scripts' / script).read_text())
            fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                      and n.name == 'hd_target_size')
            ns = dict(math=math, PATCH=16, FACTOR=32)
            exec(compile(ast.Module(body=[fn], type_ignores=[]), script, 'exec'), ns)
            image = type('Image', (), {'width':4096, 'height':4096})()
            # 16x16 UNMERGED patches = 256px edge; scale=3 => 768px.
            self.assertEqual(ns['hd_target_size'](image, [1,16,16], 3, 5017600), (768,768))
            image.width = 8192
            self.assertEqual(ns['hd_target_size'](image, [1,16,32], 3, 5017600), (1536,768))

    def test_training_units_and_seed_before_construction(self):
        tree = ast.parse(TRAIN.read_text())
        dataset = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                       and n.name == 'Qwen2VLCoupledDATDataset')
        for name in ('lr_h','lr_w'):
            assignment = next(n for n in ast.walk(dataset) if isinstance(n, ast.Assign)
                              and any(isinstance(t, ast.Name) and t.id==name for t in n.targets))
            self.assertEqual(assignment.value.right.attr, 'PATCH_SIZE')
        train = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='train')
        seed = next(n.lineno for n in ast.walk(train) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute) and n.func.attr=='set_seed')
        construction = min(n.lineno for n in ast.walk(train) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name) and n.func.id in ('convert_fn','_load_base_vlm'))
        self.assertLess(seed, construction)


if __name__ == '__main__':
    unittest.main()
