"""Independent CPU oracle and an explicitly reused, verified CUDA policy backend."""
import importlib.util
import math
from pathlib import Path
import sys

import torch


@torch.no_grad()
def symbolize(patches, segment, breakpoints):
    if patches.shape[-1] % segment:
        raise ValueError("PAA segment width must divide patch width")
    centered = patches-patches.mean(-1, keepdim=True)
    scale = patches.std(-1, unbiased=False, keepdim=True).clamp_min(1e-6)
    paa = (centered/scale).reshape(*patches.shape[:-1], -1, segment).mean(-1)
    return torch.bucketize(paa.contiguous(), breakpoints).long()


def addresses(word, bins=256):
    result = []
    for h in range(3):
        code = 17+31*h
        for symbol in word:
            code = (code*(131+2*h)+symbol+1) % 2147483647
        result.append(code % bins)
    return result


class ReferencePolicy:
    """Readable CPU-only oracle, deliberately excluded from performance timings."""
    @torch.no_grad()
    def select(self, words, ratio, generator, high_fraction=.8):
        if words.device.type != "cpu":
            raise ValueError("The independent oracle is CPU-only")
        n = words.shape[-2]
        k = max(1, round(n*ratio))
        if k == n:
            return torch.arange(n).expand(*words.shape[:-2], n)
        high = round(k*high_fraction)
        if k > 1 and 0 < high_fraction < 1:
            high = min(k-1, max(1, high))
        random = torch.rand(words.shape[:-1], generator=generator) if k > high else None
        output = []
        for row_id, tensor in enumerate(words.reshape(-1, n, words.shape[-1])):
            row = [tuple(w) for w in tensor.tolist()]
            counters, last = [0]*256, {}
            for i, word in enumerate(row):
                last[word] = i
                for h in addresses(word):
                    counters[h] += 1
            frequency = [min(counters[h] for h in addresses(w)) for w in row]
            reps = set(last.values())
            selected = set(sorted(reps, key=lambda i:(-frequency[i], i))[:high])
            if random is not None:
                scores = random.reshape(-1, n)[row_id].tolist()
                selected.update(sorted(reps-selected, key=lambda i:(-scores[i], i))[:k-high])
            priority = [(2*(i in selected)+(i in reps))*(3*n+1)+frequency[i] for i in range(n)]
            output.append(sorted(sorted(range(n), key=lambda i:(-priority[i], i))[:k]))
        return torch.tensor(output).reshape(*words.shape[:-2], k)

    @torch.no_grad()
    def route(self, words, groups, iterations=3):
        if words.device.type != "cpu":
            raise ValueError("The independent oracle is CPU-only")
        rows, n, _ = words.shape
        g, capacity = min(groups, n), math.ceil(n/min(groups, n))
        initial = torch.randperm(n, generator=torch.Generator().manual_seed(2021))[:g].tolist()
        labels, counts, slots, centers = [], [], [], []
        for row in words.tolist():
            medoids = list(initial)
            distance = lambda i,j:sum(a != b for a,b in zip(row[i], row[j]))
            for step in range(iterations+1):
                members, assigned = [[] for _ in range(g)], []
                for i in range(n):
                    group = min((j for j in range(g) if len(members[j]) < capacity),
                                key=lambda j:(distance(i, medoids[j]), j))
                    assigned.append(group)
                    members[group].append(i)
                if step < iterations:
                    for j, indices in enumerate(members):
                        if indices:
                            medoids[j] = min(indices, key=lambda i:(sum(distance(i,h) for h in indices), i))
            positions = [0]*n
            for j, indices in enumerate(members):
                for rank, i in enumerate(indices):
                    positions[i] = j*capacity+rank
            labels.append(assigned)
            counts.append([len(x) for x in members])
            slots.append(positions)
            centers.append(medoids)
        return dict(labels=torch.tensor(labels), counts=torch.tensor(counts), slots=torch.tensor(slots),
                    medoids=torch.tensor(centers), groups=g, capacity=capacity, input_tokens=n)


def load_module(path, name):
    if name in sys.modules:
        existing = sys.modules[name]
        if Path(existing.__file__).resolve() != Path(path).resolve():
            raise ValueError("Cannot mix different frozen policy sources")
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class FusedPolicy:
    """Reuse only the frozen discrete CUDA operators, never the old learned model."""
    def __init__(self, frozen=None, device="cuda"):
        frozen = Path(frozen) if frozen is not None else Path(__file__).parent / "kernels"
        self.selector_module = load_module(frozen/"selector.py", "spare_verified_selector")
        self.router_module = load_module(frozen/"fused_routing.py", "spare_verified_router")
        self.selector = self.selector_module.StatelessSelector(device=device)
        self.router = self.router_module.Router()

    def select(self, words, ratio, generator, high_fraction=.8):
        return self.selector(words, ratio, "hybrid", generator, high_fraction, 256, False)

    def route(self, words, groups, iterations=3):
        return self.router(words, groups, iterations)

    def compiler_counts(self):
        kernels = dict(tables=self.selector_module._tables, rank=self.selector_module._rank,
                       route=self.router_module._route)
        return {name:sum(len(v) for v in kernel.cache.values()) for name,kernel in kernels.items()}
