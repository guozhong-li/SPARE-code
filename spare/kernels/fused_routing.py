"""Shape-general matrix-free execution of the frozen capacity-constrained policy."""
import math

import torch
import triton
import triton.language as tl


@triton.jit
def _route(W, INIT, LABEL, SLOT, COUNT, MEDOID,
           N: tl.constexpr, G: tl.constexpr, CAP: tl.constexpr,
           BLOCK: tl.constexpr, GB: tl.constexpr, ITERS: tl.constexpr):
    row = tl.program_id(0)
    x = tl.arange(0, BLOCK)
    g = tl.arange(0, GB)
    z = tl.load(W + row*N*4 + x*4, x < N, 0).to(tl.int32)
    z |= tl.load(W + row*N*4 + x*4+1, x < N, 0).to(tl.int32) << 2
    z |= tl.load(W + row*N*4 + x*4+2, x < N, 0).to(tl.int32) << 4
    z |= tl.load(W + row*N*4 + x*4+3, x < N, 0).to(tl.int32) << 6
    medoid = tl.load(INIT + g, g < G, 0).to(tl.int32)
    labels = tl.full((BLOCK,), -1, tl.int32)
    sizes = tl.full((GB,), 0, tl.int32)
    for iteration in range(ITERS+1):
        centers = tl.load(W + row*N*4 + medoid*4, g < G, 0).to(tl.int32)
        centers |= tl.load(W + row*N*4 + medoid*4+1, g < G, 0).to(tl.int32) << 2
        centers |= tl.load(W + row*N*4 + medoid*4+2, g < G, 0).to(tl.int32) << 4
        centers |= tl.load(W + row*N*4 + medoid*4+3, g < G, 0).to(tl.int32) << 6
        sizes = tl.full((GB,), 0, tl.int32)
        labels = tl.full((BLOCK,), -1, tl.int32)
        for i in range(N):
            token = tl.load(W + row*N*4 + i*4).to(tl.int32)
            token |= tl.load(W + row*N*4 + i*4+1).to(tl.int32) << 2
            token |= tl.load(W + row*N*4 + i*4+2).to(tl.int32) << 4
            token |= tl.load(W + row*N*4 + i*4+3).to(tl.int32) << 6
            xor = token ^ centers
            distance = ((xor & 3) != 0).to(tl.int32)
            distance += (((xor >> 2) & 3) != 0).to(tl.int32)
            distance += (((xor >> 4) & 3) != 0).to(tl.int32)
            distance += (((xor >> 6) & 3) != 0).to(tl.int32)
            choice = tl.min(tl.where((g < G) & (sizes < CAP), distance*G+g, 1048576), 0) % G
            sizes += (g == choice).to(tl.int32)
            labels = tl.where(x == i, choice, labels)
        if iteration < ITERS:
            matches = tl.full((BLOCK,), 0, tl.int32)
            for coordinate in tl.static_range(4):
                symbol = (z >> (2*coordinate)) & 3
                for group in tl.static_range(G):
                    for value in tl.static_range(4):
                        member = (x < N) & (labels == group) & (symbol == value)
                        frequency = tl.sum(member.to(tl.int32), 0)
                        matches += tl.where(member, frequency, 0)
            for group in tl.static_range(G):
                member = (x < N) & (labels == group)
                best = tl.max(tl.where(member, matches, -1), 0)
                position = tl.min(tl.where(member & (matches == best), x, 1048576), 0)
                medoid = tl.where((g == group) & (position < N), position, medoid)
    slots = tl.full((BLOCK,), 0, tl.int32)
    for group in tl.static_range(G):
        rank = tl.cumsum(((x < N) & (labels == group)).to(tl.int32), 0)-1
        slots = tl.where(labels == group, group*CAP+rank, slots)
    tl.store(LABEL + row*N+x, labels, x < N)
    tl.store(SLOT + row*N+x, slots, x < N)
    tl.store(COUNT + row*G+g, sizes, g < G)
    tl.store(MEDOID + row*G+g, medoid, g < G)


class Router:
    def __init__(self):
        self.initial = {}

    @torch.no_grad()
    def __call__(self, words, groups=4, iterations=3):
        b, n, width = words.shape
        if not words.is_cuda or width != 4 or words.dtype != torch.int64 or not 1 <= n <= 256:
            raise ValueError("CUDA int64 four-symbol words; 1 <= N <= 256")
        g = min(groups, n)
        if not 1 <= g <= 4 or iterations != 3:
            raise ValueError("G <= 4 and three updates followed by final assignment")
        key = (n, g, str(words.device))
        if key not in self.initial:
            self.initial[key] = torch.randperm(n, generator=torch.Generator().manual_seed(2021))[:g].to(words.device)
        cap = math.ceil(n/g)
        out = {name: torch.empty(b, size, device=words.device, dtype=torch.long)
               for name, size in (("labels", n), ("slots", n), ("counts", g), ("medoids", g))}
        out.update(groups=g, capacity=cap, input_tokens=n)
        _route[(b,)](words.contiguous(), self.initial[key], out["labels"], out["slots"],
                     out["counts"], out["medoids"], n, g, cap,
                     triton.next_power_of_2(n), triton.next_power_of_2(g), iterations,
                     num_warps=4, enable_fp_fusion=False)
        return out
