"""Traffic execution candidates; no width, depth, selection or grouping change."""
from contextlib import contextmanager
import types
import math

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def singleton_forward(self, x, route):
    if route["capacity"] != 1:
        return self._original_forward(x, route)
    b, length, width = x.shape
    n, heads, dim = length-1, self.heads, width//self.heads
    projected = self.qkv(x).reshape(b, length, 3, heads, dim)
    q, k, v = projected.unbind(2)
    local = v[:, :-1]
    probability = self.probability if self.training else 0.
    if probability:
        # A one-key softmax is exactly one; retain independent per-head dropout.
        mask = F.dropout(torch.ones(b*route["groups"], heads, 1, 1, device=x.device, dtype=x.dtype),
                         probability, training=True).reshape(b, route["groups"], heads, 1)
        local = local * mask.gather(1, route["labels"][:, :, None, None].expand(b, n, heads, 1))
    sums = local.new_zeros(b, route["groups"], heads, dim).scatter_add(
        1, route["labels"][:, :, None, None].expand_as(local), local)
    means = sums/route["denominator"]
    mixed = self.mixer(means.flatten(-2)).reshape_as(means)
    local = local + mixed.gather(1, route["labels"][:, :, None, None].expand_as(local))
    global_q, global_k, global_v = q[:, -1:].transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    if self.backend == "math":
        weights = (global_q @ global_k.transpose(-1, -2) / math.sqrt(dim)).softmax(-1)
        global_out = (F.dropout(weights, probability, training=probability > 0) @ global_v).transpose(1, 2)
    else:
        global_out = F.scaled_dot_product_attention(global_q, global_k, global_v,
                                                   dropout_p=probability).transpose(1, 2)
    return self.output(torch.cat((local, global_out), 1).reshape(b, length, width))


@contextmanager
def install(net, mode):
    if mode not in ("native", "dense", "dense_checkpoint", "dense_singleton", "dense_singleton_checkpoint",
                    "all_math", "all_math_checkpoint", "all_math_singleton", "all_math_singleton_checkpoint",
                    "all_math_full_checkpoint"):
        raise ValueError(mode)
    originals = []
    for layer in net.layers:
        cross = layer.cross
        originals.append((cross, "backend", cross.backend))
        if mode != "native":
            cross.backend = "math"
        if mode.startswith("all_math"):
            originals.append((layer.grouped, "backend", layer.grouped.backend))
            layer.grouped.backend = "math"
        if "checkpoint" in mode and "full_checkpoint" not in mode:
            original = cross.forward

            def forward(self, x, context, function=original):
                if not self.training or not torch.is_grad_enabled():
                    return function(x, context)
                return checkpoint(function, x, context, use_reentrant=False, preserve_rng_state=True)

            originals.append((cross, "forward", original))
            cross.forward = types.MethodType(forward, cross)
        if "full_checkpoint" in mode:
            original = layer.forward

            def encode(self, x, context, route, function=original):
                if not self.training or not torch.is_grad_enabled():
                    return function(x, context, route)
                return checkpoint(function, x, context, route, use_reentrant=False, preserve_rng_state=True)

            originals.append((layer, "forward", original))
            layer.forward = types.MethodType(encode, layer)
        if "singleton" in mode:
            grouped = layer.grouped
            originals.append((grouped, "forward", grouped.forward))
            grouped._original_forward = grouped.forward
            grouped.forward = types.MethodType(singleton_forward, grouped)
    try:
        yield
    finally:
        for module, name, original in reversed(originals):
            setattr(module, name, original)
            if name == "forward" and hasattr(module, "_original_forward"):
                del module._original_forward
