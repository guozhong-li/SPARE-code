"""Stateless exact selector for four-symbol, alphabet-four SAX words.

CBF estimates (including collisions), last-occurrence representatives, stable
ties, original Torch random-number consumption and chronological output match
the frozen hybrid/top selector. No result is retained between requests.
"""
import math

import torch
import triton
import triton.language as tl


def hash_lut():
    result = []
    for packed in range(256):
        word = [(packed >> (2*j)) & 3 for j in range(4)]
        hashes = []
        for h in range(3):
            code = 17+31*h
            for symbol in word:
                code = (code*(131+2*h)+symbol+1) % 2147483647
            hashes.append(code % 256)
        result.append(hashes)
    return result


@triton.jit
def _tables(W, LUT, TABLE, CODES, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    a = tl.load(W+row*N*4+i*4, i < N, 0).to(tl.int32)
    b = tl.load(W+row*N*4+i*4+1, i < N, 0).to(tl.int32)
    c = tl.load(W+row*N*4+i*4+2, i < N, 0).to(tl.int32)
    d = tl.load(W+row*N*4+i*4+3, i < N, 0).to(tl.int32)
    code = a+(b << 2)+(c << 4)+(d << 6)
    tl.store(CODES+row*N+i, code, i < N)
    h0 = tl.load(LUT+code*3)
    h1 = tl.load(LUT+code*3+1)
    h2 = tl.load(LUT+code*3+2)
    tl.atomic_add(TABLE+row*512+h0, 1, i < N)
    tl.atomic_add(TABLE+row*512+h1, 1, i < N)
    tl.atomic_add(TABLE+row*512+h2, 1, i < N)
    tl.atomic_max(TABLE+row*512+256+code, i+1, i < N)


@triton.jit
def _statistics(LUT, TABLE, CODES, FREQ, REP, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    code = tl.load(CODES+row*N+i, i < N, 0)
    h0 = tl.load(LUT+code*3)
    h1 = tl.load(LUT+code*3+1)
    h2 = tl.load(LUT+code*3+2)
    f = tl.minimum(tl.load(TABLE+row*512+h0),
                   tl.minimum(tl.load(TABLE+row*512+h1), tl.load(TABLE+row*512+h2)))
    representative = tl.load(TABLE+row*512+256+code) == i+1
    tl.store(FREQ+row*N+i, f, i < N)
    tl.store(REP+row*N+i, representative, i < N)


@triton.jit
def _rank(LUT, TABLE, CODES, RANDOM, OUT, DEBUG, N: tl.constexpr,
          K: tl.constexpr, HIGH: tl.constexpr, BLOCK: tl.constexpr, DIAGNOSTIC: tl.constexpr):
    row = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    code = tl.load(CODES+row*N+i, i < N, 0)
    h0 = tl.load(LUT+code*3)
    h1 = tl.load(LUT+code*3+1)
    h2 = tl.load(LUT+code*3+2)
    freq = tl.minimum(tl.load(TABLE+row*512+h0),
                      tl.minimum(tl.load(TABLE+row*512+h1), tl.load(TABLE+row*512+h2)))
    rep = (tl.load(TABLE+row*512+256+code) == i+1) & (i < N)
    chosen = tl.full((BLOCK,), False, tl.int1)
    if HIGH > 0:
        # Unique integer keys make stable descending ties select earlier indices.
        key = (rep.to(tl.int64)*(3*N+1)+freq.to(tl.int64))*(N+1)+(N-1-i)
        key = tl.where(i < N, key, -1)
        ordered = tl.sort(key, descending=True)
        threshold = tl.min(tl.where(i < HIGH, ordered, 0x7fffffffffffffff), 0)
        chosen = rep & (key >= threshold)
    if K > HIGH:
        available = rep & ~chosen
        random = tl.load(RANDOM+row*N+i, i < N, 0.)
        # Nonnegative finite FP32 randoms have order-preserving unsigned bits.
        bits = random.to(tl.uint32, bitcast=True).to(tl.int64)
        key = available.to(tl.int64)*0x1000000000000+bits*(N+1)+(N-1-i)
        key = tl.where(i < N, key, -1)
        ordered = tl.sort(key, descending=True)
        threshold = tl.min(tl.where(i < K-HIGH, ordered, 0x7fffffffffffffff), 0)
        chosen = chosen | (available & (key >= threshold))
    priority = (2*chosen.to(tl.int64)+rep.to(tl.int64))*(3*N+1)+freq
    key = tl.where(i < N, priority*(N+1)+(N-1-i), -1)
    ordered = tl.sort(key, descending=True)
    index = N-1-(ordered % (N+1))
    chronological = tl.sort(tl.where(i < K, index, N), descending=False)
    tl.store(OUT+row*K+i, chronological, i < K)
    if DIAGNOSTIC:
        tl.store(DEBUG+row*3*N+i, freq, i < N)
        tl.store(DEBUG+row*3*N+N+i, rep.to(tl.int32), i < N)
        tl.store(DEBUG+row*3*N+2*N+i, chosen.to(tl.int32), i < N)


class StatelessSelector:
    def __init__(self, device='cuda'):
        # Immutable model setup, not request-specific state or a window cache.
        self.lut = torch.tensor(hash_lut(), dtype=torch.int32, device=device)

    def tables(self, words):
        n = words.shape[-2]
        flat = words.reshape(-1, n, 4).contiguous()
        rows = flat.shape[0]
        table = torch.zeros((rows, 512), dtype=torch.int32, device=words.device)
        codes = torch.empty((rows, n), dtype=torch.int32, device=words.device)
        _tables[(rows,)](flat, self.lut, table, codes, n, triton.next_power_of_2(n), num_warps=4)
        return table, codes

    @torch.no_grad()
    def __call__(self, words, ratio, policy, generator, high_fraction=.8, bins=256,
                 exact=False, diagnostic=False, torch_ranking=False, random_override=None):
        n = words.shape[-2]
        if (not words.is_cuda or words.dtype != torch.int64 or words.shape[-1] != 4
                or bins != 256 or exact or policy not in ('hybrid', 'top')
                or not 0 < ratio <= 1 or not 0 <= high_fraction <= 1 or not 1 <= n <= 4096):
            raise ValueError('Scope: CUDA int64 alphabet-4/word-4, CBF256, hybrid/top, N<=4096')
        k = max(1, round(n*ratio))
        if k == n:
            if diagnostic:
                raise ValueError('Full retention bypass has no frequency diagnostics')
            return torch.arange(n, device=words.device).expand(*words.shape[:-2], n)
        table, codes = self.tables(words)
        rows = math.prod(words.shape[:-2])
        block = triton.next_power_of_2(n)
        if torch_ranking:
            from legacy_engine import select_with_statistics
            if policy != 'hybrid' or diagnostic or random_override is not None:
                raise ValueError('Torch-ranking ablation is hybrid only')
            freq = torch.empty(words.shape[:-1], device=words.device, dtype=torch.float32)
            rep = torch.empty(words.shape[:-1], device=words.device, dtype=torch.bool)
            _statistics[(rows,)](self.lut, table, codes, freq, rep, n, block, num_warps=4)
            return select_with_statistics(words, ratio, generator, high_fraction, freq, rep)
        high = k if policy == 'top' else round(k*high_fraction)
        if policy == 'hybrid' and k > 1 and 0 < high_fraction < 1:
            high = min(k-1, max(1, high))
        random = codes
        if k > high:
            random = (torch.rand(words.shape[:-1], device=words.device, generator=generator)
                      if random_override is None else random_override.contiguous())
        out = torch.empty((*words.shape[:-2], k), device=words.device, dtype=torch.int64)
        debug = (torch.empty((rows, 3, n), device=words.device, dtype=torch.int32)
                 if diagnostic else out)
        _rank[(rows,)](self.lut, table, codes, random, out, debug, n, k, high, block,
                      diagnostic, num_warps=4)
        return (out, debug.reshape(*words.shape[:-2], 3, n)) if diagnostic else out


def compiler_counts():
    return {name: sum(len(v) for v in kernel.cache.values())
            for name, kernel in (('tables', _tables), ('statistics', _statistics), ('rank', _rank))}
