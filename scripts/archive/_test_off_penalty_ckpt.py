#!/usr/bin/env python3
"""Does the out-of-range penalty survive gradient checkpointing?

The penalty is injected with register_hook inside the attention forward, and
training runs with gradient_checkpointing + use_reentrant=False. Under
checkpointing the forward runs twice: once under no_grad to produce activations,
once during backward to rebuild the graph. If the hook only registered on the
first pass it would silently contribute nothing, and the run would look healthy
for nine hours while the penalty did nothing at all.

Compares the checkpointed gradient against the non-checkpointed one.
"""

import torch
from torch.utils.checkpoint import checkpoint

W = 2.0
torch.manual_seed(0)
x_in = torch.randn(64)
scale0 = torch.tensor(1.6)          # pushes ~40% of entries out of range


def block(inp, scale, use_penalty):
    x = inp * scale
    if use_penalty and x.requires_grad:
        over = ((x.abs() - 1.0).clamp_min(0) * torch.sign(x)).detach()
        coef = W * 2.0 / x.numel()
        x.register_hook(lambda g, o=over, c=coef: g + c * o)
    return (x.clamp(-1, 1) * torch.linspace(0.5, 1.5, x.numel())).sum()


def run(use_ckpt, use_penalty):
    s = scale0.clone().requires_grad_(True)
    if use_ckpt:
        out = checkpoint(block, x_in, s, use_penalty, use_reentrant=False)
    else:
        out = block(x_in, s, use_penalty)
    out.backward()
    return s.grad.item()


plain_off = run(False, False)
plain_on = run(False, True)
ckpt_off = run(True, False)
ckpt_on = run(True, True)

print("no checkpointing : penalty off = %.6f   penalty on = %.6f   (delta %.6f)"
      % (plain_off, plain_on, plain_on - plain_off))
print("checkpointing    : penalty off = %.6f   penalty on = %.6f   (delta %.6f)"
      % (ckpt_off, ckpt_on, ckpt_on - ckpt_off))

survives = abs((ckpt_on - ckpt_off) - (plain_on - plain_off)) < 1e-6
active = abs(ckpt_on - ckpt_off) > 1e-9
print()
print("penalty active under checkpointing : %s" % active)
print("matches non-checkpointed gradient  : %s" % survives)
if not (active and survives):
    raise SystemExit("FAIL: penalty does not survive gradient checkpointing")
print("OK")
