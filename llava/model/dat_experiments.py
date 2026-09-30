"""Explicit controls for Qwen3.5 DAT sampler experiments.

Keys are (layer, batch row, image index); coordinates retain every reader slot
and offset group: [slots, groups, grid_h, grid_w, xy]. Controls are scoped to a
single batch. Reuse the replay scope through backward when checkpointing.
"""
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
from torch import nn

LOCAL_SAMPLER_MODULES = (
    'conv_lr_dw', 'ln_1', 'conv_lr_proj', 'proj_intention', 'ln_2', 'conv_off_proj',
)


def sampler_dat_config(checkpoint_dat, local_query_pos):
    """Keep checkpoint architecture/readers; only enable the controlled factors."""
    if not checkpoint_dat or checkpoint_dat.get('grid_size') != 20:
        raise ValueError('sampler-only requires a full DAT checkpoint with grid_size=20')
    if local_query_pos not in ('im_start', 'ans_prev'):
        raise ValueError(f'unknown local query: {local_query_pos}')
    dat = dict(checkpoint_dat)
    dat.update(local_query_pos=local_query_pos, hd_position_mode='slot',
        lr_drop_prob=0., hd_lse_bias=0., off_penalty=0., off_sup_weight=0.,
        rel_sup_weight=0., tf_force_prob=0., route_by_lr_drop=False)
    return dat


def dat_attention_modules(model):
    return [m for m in model.modules() if hasattr(m, '_dat_sampling_control')]


def freeze_local_sampler_only(model, *, zero_init=True):
    """Final freeze pass. Frozen readers still differentiate their inputs."""
    model.requires_grad_(False)
    modules = dat_attention_modules(model)
    if not modules:
        raise ValueError('sampler-only requires Qwen3.5 DAT attention modules')
    for attention in modules:
        if attention.grid_size != 20:
            raise ValueError('controlled experiments require 20x20 = 400 tokens per reader')
        for name in LOCAL_SAMPLER_MODULES:
            getattr(attention, name).requires_grad_(True)
        if zero_init:
            with torch.no_grad():
                attention.conv_off_proj.weight.zero_()
        attention.hd_position_mode = 'slot'
    # Make frozen reference / student runs deterministic in train and eval modes.
    disable_dropout(model)
    return [name for name, p in model.named_parameters() if p.requires_grad]


def disable_dropout(model):
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
        if hasattr(module, 'attention_dropout'):
            module.attention_dropout = 0.0


def assert_sampler_optimizer(model, optimizer):
    expected = {id(p): name for name, p in model.named_parameters() if p.requires_grad}
    actual = [id(p) for group in optimizer.param_groups for p in group['params']]
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise RuntimeError('optimizer must contain each local sampler parameter exactly once')
    for name in expected.values():
        components = name.split('.')
        if 'self_attn' not in components or not any(k in components for k in LOCAL_SAMPLER_MODULES):
            raise RuntimeError(f'non-local parameter is trainable: {name}')
    return list(expected.values())


def require_pure_ce(model, *, exact=True):
    dat = model.config.dat_extra_args
    for name in ('lr_drop_prob', 'hd_lse_bias', 'off_penalty', 'off_sup_weight', 'rel_sup_weight'):
        if float(dat.get(name, 0) or 0) != 0:
            raise ValueError(f'pure CE requires {name}=0')
    for module in dat_attention_modules(model):
        for name in ('off_penalty', 'off_sup_weight', 'rel_sup_weight', 'hd_lse_bias'):
            if float(getattr(module, name, 0)) != 0:
                raise ValueError(f'pure CE requires {name}=0 on layer {module.layer_idx}')
        if exact and (module.hd_gate is not None or not module._dat_exact_merge_available):
            raise ValueError('pure CE coordinate checks require exact merge backward and hd_gate=None')


def configure_pure_ce(model):
    """Disable checkpoint training curricula/hooks for a diagnostic forward."""
    dat = model.config.dat_extra_args
    for name in ('lr_drop_prob', 'hd_lse_bias', 'off_penalty', 'off_sup_weight', 'rel_sup_weight'):
        dat[name] = 0.0
    dat['tf_force_prob'] = 0.0
    dat['route_by_lr_drop'] = False
    for module in dat_attention_modules(model):
        for name in ('off_penalty', 'off_sup_weight', 'rel_sup_weight', 'hd_lse_bias'):
            setattr(module, name, 0.0)
        module.tf_force_prob = 0.0
        module.route_by_lr_drop = False
        module._dat_force_locs = module._dat_force_batch = None
        module._dat_off_target = module._dat_off_target_batch = None
    disable_dropout(model)


@dataclass
class DATSamplingControl:
    mode: str = 'capture'
    coarse: dict = field(default_factory=dict)
    overrides: dict = field(default_factory=dict)
    records: dict = field(default_factory=dict)
    zero_residual: bool = False
    observe: bool = False

    def key(self, attention):
        return (attention.layer_idx, *attention._dat_sample_key)

    def resolve_coarse(self, attention, current):
        key = self.key(attention)
        if self.mode == 'capture':
            self.coarse[key] = current.detach().clone()
            return current
        if self.mode != 'replay':
            raise ValueError(f'unknown coarse control mode: {self.mode}')
        if key not in self.coarse:
            raise KeyError(f'no reference coarse grid for {key}')
        grid = self.coarse[key].to(device=current.device, dtype=current.dtype)
        if grid.shape != current.shape:
            raise ValueError(f'coarse shape differs for {key}: {grid.shape} vs {current.shape}')
        return grid

    def override(self, attention, coordinates, slots):
        key = self.key(attention)
        if key not in self.overrides:
            return coordinates
        value = self.overrides[key]
        shape = (slots, attention.off_grps, attention.grid_size, attention.grid_size, 2)
        if tuple(value.shape) != shape:
            raise ValueError(f'coordinate override for {key} must have shape {shape}; got {value.shape}')
        # Keep leaf-to-reader gradients. No averaging or broadcasting groups.
        return value.to(device=coordinates.device, dtype=torch.float32).reshape(
            slots * attention.off_grps, attention.grid_size, attention.grid_size, 2
        ).permute(0, 3, 1, 2)

    def record(self, attention, pre_clamp, sampled, raw_offsets, offsets, hd_shape, slots, n_unsup_lead):
        if not self.observe:
            return
        for value in (pre_clamp, sampled):
            if value.requires_grad:
                value.retain_grad()
        self.records[self.key(attention)] = dict(
            pre_clamp=pre_clamp, sampled=sampled, raw_offsets=raw_offsets.detach(),
            residual=offsets.detach(),
            hd_shape=tuple(hd_shape), slots=slots, groups=attention.off_grps,
            leading_image_slots=n_unsup_lead, off_range=attention.off_range,
        )

    def replay(self, *, observe=False, overrides=None, zero_residual=False):
        return DATSamplingControl(mode='replay', coarse=self.coarse,
            observe=observe, overrides={} if overrides is None else overrides,
            zero_residual=zero_residual)


@contextmanager
def sampling_control(model, control):
    modules = dat_attention_modules(model)
    if not modules:
        raise ValueError('sampling controls require Qwen3.5 DAT attention modules')
    previous = [m._dat_sampling_control for m in modules]
    try:
        for module in modules:
            module._dat_sampling_control = control
        yield control
    finally:
        for module, old in zip(modules, previous):
            module._dat_sampling_control = old


_GRAD_UNSET = object()


def coordinate_summary(record, gradient=_GRAD_UNSET):
    """Report each slot/group independently; 8 groups still make 400 tokens."""
    slots, groups = record['slots'], record['groups']
    sampled = record['sampled']
    pre = record['pre_clamp'].permute(0, 2, 3, 1).detach()
    raw = record['raw_offsets'].permute(0, 2, 3, 1)
    residual = record['residual'].permute(0, 2, 3, 1)
    gradient = sampled.grad if gradient is _GRAD_UNSET else gradient
    summaries = []
    for slot in range(slots):
        for group in range(groups):
            row = slot * groups + group
            locs = sampled[row].detach()
            g = None if gradient is None else gradient[row].detach().float()
            summaries.append(dict(slot=slot, group=group,
                reader='image' if slot < record['leading_image_slots'] else 'answer',
                tokens=int(locs.shape[0] * locs.shape[1]),
                grad_norm=None if g is None else float(g.norm()),
                nonzero_fraction=None if g is None else float((g != 0).float().mean()),
                pre_clamp_oob_fraction=float((pre[row].abs() > 1).float().mean()),
                boundary_fraction=float((locs.abs() >= 1).any(-1).float().mean()),
                raw_residual_min=float(raw[row].min()), raw_residual_max=float(raw[row].max()),
                residual_min=float(residual[row].min()), residual_max=float(residual[row].max()),
                tanh_saturation_fraction=(float((raw[row].tanh().abs() > .99).float().mean())
                                          if record['off_range'] > 0 else 0.0)))
    return summaries
