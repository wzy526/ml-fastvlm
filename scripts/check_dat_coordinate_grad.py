#!/usr/bin/env python3
"""Pure answer CE gradients and central differences on per-group DAT coordinates.

Requires a full Qwen3.5 DAT checkpoint and CUDA exact-merge backend. The model
is frozen; only coordinates require gradients. No optimizer or auxiliary loss.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def ce_parts(logits, labels, template_ids, newline_ids=()):
    """Shift labels once and separate content, template and first content CE."""
    targets = labels[:, 1:]
    valid = targets != -100
    is_template = torch.zeros_like(valid)
    for token in template_ids:
        is_template |= targets == token
    # Newlines inside an answer are content. Only newlines following its end
    # marker in the same supervised span belong to the trailing template.
    indices = torch.arange(targets.shape[-1], device=targets.device).expand_as(targets)
    last_end = torch.where(is_template, indices, -1).cummax(-1).values
    last_gap = torch.where(~valid, indices, -1).cummax(-1).values
    for token in newline_ids:
        is_template |= (targets == token) & (last_end > last_gap)
    content = valid & ~is_template
    first = torch.zeros_like(valid)
    for row in range(labels.shape[0]):
        positions = content[row].nonzero().flatten()
        if len(positions):
            first[row, positions[0]] = True
    scores = logits[:, :-1]
    out = {}
    for name, mask in (('total', valid), ('content', content), ('template', valid & is_template), ('first', first)):
        out[name] = (F.cross_entropy(scores[mask].float(), targets[mask])
                     if mask.any() else None)
    if out['total'] is None or out['content'] is None:
        raise ValueError('sample needs at least one supervised answer content token')
    return out


def fd_candidates(coordinates, gradient, hd_shape, count):
    """Select interior points at least .15 HD cells from interpolation knots."""
    height, width = hd_shape
    scale = coordinates.new_tensor([max(width - 1, 1), max(height - 1, 1)]) / 2
    cell = (coordinates + 1) * scale
    fraction = cell - cell.floor()
    smooth = ((fraction > .15) & (fraction < .85) & (coordinates.abs() < .98)).all(-1)
    score = gradient.abs().clone()
    score[~smooth] = -1
    ranked = score.reshape(-1).argsort(descending=True)
    result = []
    for flat in ranked[:count]:
        index = []
        number = int(flat)
        for dim in reversed(coordinates.shape):
            index.append(number % dim)
            number //= dim
        index = tuple(reversed(index))
        if float(score[index]) < 0:
            continue
        result.append(index)
    return result


def central_difference(evaluate, coordinates, index, cell_step, hd_shape):
    axis = index[-1]
    edge = hd_shape[1] if axis == 0 else hd_shape[0]
    if edge <= 1:
        raise ValueError('finite differences require an HD map with both edges > 1')
    epsilon = 2.0 * cell_step / (edge - 1)
    plus, minus = coordinates.detach().clone(), coordinates.detach().clone()
    plus[index] += epsilon
    minus[index] -= epsilon
    upper, lower = float(evaluate(plus)), float(evaluate(minus))
    return dict(cell_step=cell_step, normalized_step=epsilon,
                plus_ce=upper, minus_ce=lower, loss_delta=upper-lower,
                finite_difference=(upper-lower)/(2*epsilon))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model_path', required=True)
    ap.add_argument('--reference_model_path', help='Initial frozen checkpoint used by sampler-only training; defaults to model_path')
    ap.add_argument('--data_json', required=True)
    ap.add_argument('--image_folder', required=True)
    ap.add_argument('--processor_path')
    ap.add_argument('--n', type=int, default=16)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--local_query_pos', choices=['im_start','ans_prev'], default='ans_prev')
    ap.add_argument('--residual', choices=['zero','learned'], default='zero')
    ap.add_argument('--fd_points', type=int, default=2, help='Per layer/image, across all slots and groups')
    ap.add_argument('--fd_steps', type=float, nargs='+', default=[.01,.05,.1])
    ap.add_argument('--tok_budget', type=int, default=256)
    ap.add_argument('--min_pixels', type=int, default=28224)
    ap.add_argument('--hd_cap', type=int, default=5017600)
    ap.add_argument('--hr_scale', type=int, default=3)
    ap.add_argument('--max_answer_tokens', type=int, default=128)
    ap.add_argument('--out', required=True)
    ap.add_argument('--coordinates_in', help='JSON coordinate overrides from coordinates_out; indexed by sample and (layer,batch,image)')
    ap.add_argument('--coordinates_out', help='Save complete per-slot/per-group coordinates for replay')
    args = ap.parse_args()
    if args.n < 1 or args.fd_points < 0 or any(x <= 0 or x > .1 for x in args.fd_steps):
        ap.error('n >= 1, fd_points >= 0 and 0 < fd_steps <= .1 are required')
    # Set before importing the model/backend. Keep FD forwards on the same
    # exact branch as the analytic forward (grad enabled, coordinate leaves).
    os.environ['DAT_EXACT_MERGE_GRAD'] = '1'
    from transformers import AutoProcessor, set_seed
    from llava.model.language_model.modeling_qwen3_5_dat import Qwen3_5DATForConditionalGeneration
    from llava.model.dat_experiments import (
        DATSamplingControl, sampling_control, dat_attention_modules,
        coordinate_summary, require_pure_ce, configure_pure_ce, LOCAL_SAMPLER_MODULES,
    )
    from scripts._test_lr_drop_leverage import load_items, build
    set_seed(args.seed)
    device = 'cuda:0'
    model = Qwen3_5DATForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype='auto', device_map={'':device}).eval().requires_grad_(False)
    if args.reference_model_path and args.reference_model_path != args.model_path:
        reference = Qwen3_5DATForConditionalGeneration.from_pretrained(
            args.reference_model_path, torch_dtype='auto', device_map={'':device}).eval().requires_grad_(False)
    else:
        reference = model
    for candidate in {model, reference}:
        configure_pure_ce(candidate)
        require_pure_ce(candidate)
        if candidate.config.dat_extra_args['grid_size'] != 20:
            raise ValueError('checkpoint must use 20x20 = 400 tokens per reader')
        for module in dat_attention_modules(candidate):
            module.hd_position_mode = 'slot'
    if reference is not model:
        ref_params = dict(reference.named_parameters())
        for name,param in model.named_parameters():
            parts = name.split('.')
            if 'self_attn' in parts and any(part in parts for part in LOCAL_SAMPLER_MODULES):
                continue
            if name not in ref_params or not torch.equal(param,ref_params[name]):
                raise ValueError(f'reader/coarse parameter differs from frozen reference: {name}')
    ppath = args.processor_path or args.model_path
    processor = AutoProcessor.from_pretrained(ppath, min_pixels=args.min_pixels,
        max_pixels=args.tok_budget*1024, use_fast=False)
    hr_processor = AutoProcessor.from_pretrained(ppath, min_pixels=1024,
        max_pixels=100_000_000, use_fast=False)
    template_ids = set(processor.tokenizer.encode('<|im_end|>', add_special_tokens=False))
    newline_ids = set(processor.tokenizer.encode('\n', add_special_tokens=False))
    args.oracle_min_cells = 20  # input builder's unused oracle geometry
    results = []
    saved_coordinates = []
    imported_coordinates = (json.loads(Path(args.coordinates_in).read_text())['coordinates']
                            if args.coordinates_in else [])
    for sample_index, item in enumerate(load_items(args.data_json, args.image_folder, args.n, args.seed)):
        base, hd, _, _, _, _ = build(item, sample_index, processor, hr_processor, args,
                                      device, model.dtype, 20)
        inputs = dict(base, **hd, use_cache=False, return_dict=True)
        capture = DATSamplingControl(zero_residual=True)
        for module in dat_attention_modules(reference):
            module.local_query_pos = 'im_start'
        with sampling_control(reference, capture), torch.no_grad():
            reference(**inputs)
        for module in dat_attention_modules(model):
            module.local_query_pos = args.local_query_pos
        baseline = capture.replay(observe=True, zero_residual=args.residual=='zero')
        with sampling_control(model, baseline), torch.no_grad():
            model(**inputs)
        if not baseline.records:
            raise RuntimeError('sample did not reach a DAT reader')
        overrides = {key: record['sampled'].detach().reshape(
            record['slots'], record['groups'],20,20,2).clone().requires_grad_(True)
            for key,record in baseline.records.items()}
        for entry in imported_coordinates:
            if entry['sample_index'] != sample_index:
                continue
            key = tuple(entry['key'])
            if key not in overrides:
                raise ValueError(f'unknown coordinate override key {key} for sample {sample_index}')
            overrides[key] = torch.tensor(entry['coordinates'], device=device,
                                           dtype=torch.float32, requires_grad=True)
        if args.coordinates_out:
            saved_coordinates.extend(dict(sample_index=sample_index,key=list(key),
                coordinates=value.detach().cpu().tolist()) for key,value in overrides.items())
        control = capture.replay(observe=True, overrides=overrides)
        with sampling_control(model, control):
            outputs = model(**inputs)
            losses = ce_parts(outputs.logits, inputs['labels'], template_ids, newline_ids)
            records = list(control.records.items())
            tensors = [record['sampled'] for _,record in records]
            gradients = {name: torch.autograd.grad(loss, tensors, retain_graph=True,
                         allow_unused=True) for name,loss in losses.items() if loss is not None}
            leaf_grads = torch.autograd.grad(losses['total'], list(overrides.values()),
                                            allow_unused=False)
        branches = {str(m.layer_idx):getattr(m,'_dat_last_merge_branch',None)
                    for m in dat_attention_modules(model)}
        if any(branch != 'exact' for branch in branches.values()):
            raise RuntimeError(f'coordinate diagnostic requires exact merge; got {branches}')
        sample = dict(sample_index=sample_index, image=item[0], question=item[1], answer=item[2],
                      ce={k:None if v is None else float(v.detach()) for k,v in losses.items()},
                      merge_branches=branches, readers=[])
        for i,(key,record) in enumerate(records):
            grad = leaf_grads[list(overrides).index(key)]
            reader = dict(key=list(key), hd_shape=record['hd_shape'],
                gradients={name:coordinate_summary(record, values[i]) for name,values in gradients.items()},
                finite_differences=[])
            coordinate = overrides[key]
            candidates = fd_candidates(coordinate.detach(), grad.detach(), record['hd_shape'], args.fd_points)
            for index in candidates:
                analytic = float(grad[index])
                def evaluate(value):
                    changed = dict(overrides)
                    changed[key] = value.requires_grad_(True)
                    replay = capture.replay(overrides=changed)
                    with sampling_control(model, replay), torch.enable_grad():
                        out = model(**inputs)
                        return ce_parts(out.logits, inputs['labels'], template_ids, newline_ids)['total'].detach()
                for step in args.fd_steps:
                    diff = central_difference(evaluate, coordinate, index, step, record['hd_shape'])
                    estimate = diff['finite_difference']
                    diff.update(index=list(index), analytic=analytic,
                        sign_agrees=(analytic*estimate>0) if analytic and estimate else None,
                        relative_error=abs(analytic-estimate)/max(abs(analytic),abs(estimate),1e-12),
                        unresolved_loss_delta=abs(diff['loss_delta']) <= 1e-6)
                    reader['finite_differences'].append(diff)
            if not candidates:
                reader['fd_skip_reason'] = 'no interior point away from interpolation knots'
            sample['readers'].append(reader)
        results.append(sample)
        print(f"[{sample_index+1}] CE={sample['ce']['total']:.6f}; checked {len(records)} readers", flush=True)
    if not results:
        raise ValueError('no valid answer samples found')
    report = dict(config=vars(args), position_mode='slot', tokens_per_reader=400,
                  pure_ce=True, per_sample=results,
                  numerical_note='BF16 reader quantization can hide small loss deltas; inspect all steps. CPU fp32 sampler regression provides a smooth reference.')
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    if args.coordinates_out:
        Path(args.coordinates_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.coordinates_out).write_text(json.dumps(dict(config=vars(args),
            coordinates=saved_coordinates), ensure_ascii=False)+'\n')


if __name__ == '__main__':
    main()
