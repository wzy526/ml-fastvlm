#!/usr/bin/env python3
"""Check that the hook-injected out-of-range penalty equals the loss-routed one.

The model injects the penalty gradient onto x with register_hook instead of
adding a term to the loss, because under gradient checkpointing the forward that
builds the graph is the recomputed one and a stashed penalty tensor would carry
no grad_fn. That trades a readable formulation for one that is easy to get
subtly wrong, so it is checked numerically here.

Also verifies the property the penalty exists for: gradient is exactly 1 inside
the region (a plain clamp), versus tanh which damps everything.
"""

import torch

W = 1.0
torch.manual_seed(0)
x0 = torch.randn(4, 2, 6, 6) * 1.2          # ~40% of entries out of range


def downstream(y):
    """stand-in for grid_sample + the rest of the network"""
    return (y * torch.linspace(0.5, 1.5, y.numel()).reshape(y.shape)).sum()


# reference: penalty added to the loss
xa = x0.clone().requires_grad_(True)
loss_a = downstream(xa.clamp(-1, 1)) + W * (xa.abs() - 1).clamp_min(0).pow(2).mean()
loss_a.backward()

# what the model does: same penalty injected as a gradient
xb = x0.clone().requires_grad_(True)
over = ((xb.abs() - 1.0).clamp_min(0) * torch.sign(xb)).detach()
coef = W * 2.0 / xb.numel()
xb.register_hook(lambda g, o=over, c=coef: g + c * o)
loss_b = downstream(xb.clamp(-1, 1))
loss_b.backward()

same = torch.allclose(xa.grad, xb.grad, atol=1e-7)
print("hook == loss-routed penalty : %s   max|diff| = %.3e"
      % (same, (xa.grad - xb.grad).abs().max().item()))

# the pull-back points inward for every escaped point
oob = x0.abs() > 1
pull = xb.grad[oob] * torch.sign(x0[oob])
print("out-of-range grad points inward (all > 0 after sign-align): %s  min=%.4f"
      % (bool((pull > 0).all()), pull.min().item()))

# in-region gradient is untouched by the penalty; tanh would have damped it
xc = x0.clone().requires_grad_(True)
downstream(1.0 * torch.tanh(xc)).backward()
ins = ~oob
print("in-region |grad|  clamp+penalty = %.4f   tanh = %.4f   (tanh damps %.1fx)"
      % (xb.grad[ins].abs().mean(), xc.grad[ins].abs().mean(),
         xb.grad[ins].abs().mean() / xc.grad[ins].abs().mean()))
print("out-of-range |grad| tanh = %.6f  (vanishing is the problem being avoided)"
      % xc.grad[oob].abs().mean())
