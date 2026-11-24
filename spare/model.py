"""Standalone numerical model: no inheritance from or monkeypatching of old code."""
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from .policy import ReferencePolicy, symbolize


@dataclass(frozen=True)
class Config:
    length: int = 512
    horizon: int = 96
    variables: int = 21
    width: int = 512
    feedforward: int = 2048
    heads: int = 8
    layers: int = 2
    patch: int = 16
    segment: int = 4
    alphabet: int = 4
    retention: float = .5
    groups: int = 4
    dropout: float = .1
    seed: int = 2021
    activation: str = "gelu"
    normalize: bool = True

    def __post_init__(self):
        if self.heads < 1 or self.segment < 1 or self.width < 2 or self.feedforward < 1 or self.horizon < 1:
            raise ValueError("Invalid model dimensions")
        if not (self.length >= self.patch > 0 and self.patch % self.segment == 0
                and self.width % self.heads == 0 and self.width % 2 == 0
                and 0 < self.retention <= 1 and self.groups > 0 and self.layers > 0
                and self.variables > 0 and self.alphabet >= 2 and 0 <= self.dropout < 1):
            raise ValueError("Invalid model configuration")
        if self.activation not in ("relu", "gelu"):
            raise ValueError("Unsupported activation")

    @classmethod
    def from_legacy(cls, args):
        if args.features != "M":
            raise ValueError("This paper-aligned pilot supports multivariate forecasting only")
        return cls(length=args.seq_len, horizon=args.pred_len, variables=args.enc_in,
            width=args.d_model, feedforward=args.d_ff, heads=args.n_heads,
            layers=args.e_layers, patch=args.patch_len, segment=args.sax_seg_len,
            alphabet=args.sax_alphabet, retention=args.keep_ratio, groups=args.num_groups,
            dropout=args.dropout, seed=args.training_seed, activation=args.activation,
            normalize=bool(args.use_norm))


def plan(route):
    slots, groups, capacity = route["slots"], route["groups"], route["capacity"]
    valid = torch.zeros(slots.shape[0], groups*capacity, dtype=torch.bool, device=slots.device)
    valid.scatter_(1, slots, True)
    valid = valid.reshape(-1, capacity)
    safe = valid.clone()
    safe[:, 0] |= ~valid.any(-1)
    return dict(**route, valid=valid, safe=safe,
                denominator=route["counts"][:, :, None, None].clamp_min(1))


def attention(q, k, v, probability, valid=None, backend="sdpa"):
    if backend == "sdpa":
        return F.scaled_dot_product_attention(q, k, v, attn_mask=valid, dropout_p=probability)
    if backend != "math":
        raise ValueError(backend)
    scores = (q@k.transpose(-1, -2))/math.sqrt(q.shape[-1])
    if valid is not None:
        scores = scores.masked_fill(~valid, -float("inf"))
    weights = F.dropout(scores.softmax(-1), probability, training=probability > 0)
    return weights@v


class GroupedAttention(nn.Module):
    def __init__(self, cfg, backend):
        super().__init__()
        self.heads, self.width, self.probability, self.backend = cfg.heads, cfg.width, cfg.dropout, backend
        self.qkv = nn.Linear(cfg.width, 3*cfg.width)
        self.output = nn.Linear(cfg.width, cfg.width)
        self.mixer = nn.Sequential(nn.Linear(cfg.width, cfg.width), nn.GELU(),
                                   nn.Linear(cfg.width, cfg.width), nn.Dropout(cfg.dropout))

    def forward(self, x, route):
        b, length, width = x.shape
        n, heads, dim = length-1, self.heads, width//self.heads
        groups, capacity = route["groups"], route["capacity"]
        projected = self.qkv(x).reshape(b, length, 3, heads, dim)
        q, k, v = projected.unbind(2)
        probability = self.probability if self.training else 0.
        index = route["slots"][:, :, None, None].expand(b, n, heads, dim)
        if capacity == 1 and not self.training:
            local = v[:, :-1]
            sums = local.new_zeros(b, groups, heads, dim).scatter_add(
                1, route["labels"][:, :, None, None].expand_as(local), local)
            means = sums/route["denominator"]
        else:
            # Pack Q/K/V in a single scatter, keeping group-sized attention blocks.
            packed = projected.new_zeros(b, groups*capacity, 3, heads, dim).scatter(
                1, route["slots"][:, :, None, None, None].expand(b, n, 3, heads, dim), projected[:, :-1])
            packed = packed.reshape(b*groups, capacity, 3, heads, dim).permute(2, 0, 3, 1, 4)
            out = attention(*packed.unbind(0), probability, route["safe"][:, None, None, :], self.backend)
            out = (out.transpose(1, 2)*route["valid"][:, :, None, None]).reshape(b, groups, capacity, heads, dim)
            means = out.sum(2)/route["denominator"]
            local = out.reshape(b, groups*capacity, heads, dim).gather(1, index)
        mixed = self.mixer(means.flatten(-2)).reshape_as(means)
        local = local+mixed.gather(1, route["labels"][:, :, None, None].expand_as(local))
        global_out = attention(q[:, -1:].transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                               probability, backend=self.backend).transpose(1, 2)
        return self.output(torch.cat((local, global_out), 1).reshape(b, length, width))


class CrossAttention(nn.Module):
    def __init__(self, cfg, backend):
        super().__init__()
        self.heads, self.probability, self.backend = cfg.heads, cfg.dropout, backend
        self.query = nn.Linear(cfg.width, cfg.width)
        self.kv = nn.Linear(cfg.width, 2*cfg.width)
        self.output = nn.Linear(cfg.width, cfg.width)

    def forward(self, x, context):
        b, length, width = x.shape
        q = self.query(x).reshape(b, length, self.heads, -1).transpose(1, 2)
        k, v = self.kv(context).reshape(b, context.shape[1], 2, self.heads, -1).unbind(2)
        out = attention(q, k.transpose(1, 2), v.transpose(1, 2),
                        self.probability if self.training else 0., backend=self.backend)
        return self.output(out.transpose(1, 2).reshape(b, length, width))


class EncoderLayer(nn.Module):
    def __init__(self, cfg, backend):
        super().__init__()
        self.grouped = GroupedAttention(cfg, backend)
        self.cross = CrossAttention(cfg, backend)
        self.first = nn.Linear(cfg.width, cfg.feedforward)
        self.second = nn.Linear(cfg.feedforward, cfg.width)
        self.norm1, self.norm2, self.norm3 = [nn.LayerNorm(cfg.width) for _ in range(3)]
        self.dropout = nn.Dropout(cfg.dropout)
        self.activation = F.gelu if cfg.activation == "gelu" else F.relu

    def forward(self, x, context, route):
        x = self.norm1(x+self.dropout(self.grouped(x, route)))
        batch = context.shape[0]
        global_token = x[:, -1:]
        cross = self.dropout(self.cross(global_token.reshape(batch, -1, x.shape[-1]), context))
        global_token = self.norm2(global_token+cross.reshape_as(global_token))
        x = torch.cat((x[:, :-1], global_token), 1)
        y = self.dropout(self.activation(self.first(x)))
        return self.norm3(x+self.dropout(self.second(y)))


class Spare(nn.Module):
    def __init__(self, cfg, policy=None, attention_backend="sdpa"):
        super().__init__()
        self.cfg, self.policy = cfg, policy if policy is not None else ReferencePolicy()
        self.patches = cfg.length//cfg.patch
        if self.patches > 5000:
            raise ValueError("Position table supports at most 5000 patches")
        self.kept = max(1, round(self.patches*cfg.retention))
        self.patch_embedding = nn.Linear(cfg.patch, cfg.width, bias=False)
        self.global_token = nn.Parameter(torch.randn(1, cfg.variables, 1, cfg.width))
        position = torch.arange(5000).float()[:, None]
        div = (torch.arange(0, cfg.width, 2).float()*-(math.log(10000.)/cfg.width)).exp()
        pe = torch.zeros(5000, cfg.width)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(position*div), torch.cos(position*div)
        self.register_buffer("position", pe[None])
        probs = torch.linspace(0, 1, cfg.alphabet+1)[1:-1]
        self.register_buffer("breakpoints", math.sqrt(2.)*torch.erfinv(2*probs-1))
        self.context_embedding = nn.Linear(cfg.length, cfg.width)
        self.embedding_dropout = nn.Dropout(cfg.dropout)
        self.context_dropout = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList([EncoderLayer(cfg, attention_backend) for _ in range(cfg.layers)])
        self.norm = nn.LayerNorm(cfg.width)
        self.head = nn.Linear(cfg.width*(self.kept+1), cfg.horizon)
        self.head_dropout = nn.Dropout(cfg.dropout)
        self._generators = {}
        self.last_selected_indices = self.last_routing = None

    def reset_policy_rng(self):
        self._generators.clear()

    def generator(self, device):
        key = str(device)
        if key not in self._generators:
            self._generators[key] = torch.Generator(device=device).manual_seed(self.cfg.seed+100003)
        return self._generators[key]

    def train(self, mode=True):
        if self.training != mode:
            self.reset_policy_rng()
        return super().train(mode)

    def forward(self, x, marks=None, decoder=None, future_marks=None):
        cfg = self.cfg
        if x.shape[1:] != (cfg.length, cfg.variables):
            raise ValueError("Input shape disagrees with model configuration")
        if cfg.normalize:
            mean = x.mean(1, keepdim=True).detach()
            centered = x-mean
            scale = torch.sqrt(centered.var(1, keepdim=True, unbiased=False)+1e-5)
            x = centered/scale
        patches = x.transpose(1, 2).unfold(-1, cfg.patch, cfg.patch)
        words = symbolize(patches.detach(), cfg.segment, self.breakpoints)
        indices = self.policy.select(words, cfg.retention, self.generator(x.device))
        raw = patches.gather(2, indices[..., None].expand(*indices.shape, cfg.patch))
        selected_words = words.gather(2, indices[..., None].expand(*indices.shape, words.shape[-1]))
        batch, variables, kept, _ = raw.shape
        routing = self.policy.route(selected_words.reshape(batch*variables, kept, -1), cfg.groups)
        route = plan(routing)
        numerical = self.patch_embedding(raw.reshape(batch*variables, kept, cfg.patch))
        numerical = numerical+self.position[0, indices.reshape(batch*variables, kept)]
        token = self.global_token.expand(batch, -1, -1, -1).reshape(batch*variables, 1, cfg.width)
        numerical = self.embedding_dropout(torch.cat((numerical, token), 1))
        # Preserve the old backbone's exogenous-context choice for weight-matched controls.
        context = x[:, :, :-1].transpose(1, 2)
        if marks is not None:
            context = torch.cat((context, marks.transpose(1, 2)), 1)
        context = self.context_dropout(self.context_embedding(context))
        for layer in self.layers:
            numerical = layer(numerical, context, route)
        numerical = self.norm(numerical).reshape(batch, variables, kept+1, cfg.width)
        flattened = numerical.permute(0, 1, 3, 2).flatten(-2)
        prediction = self.head_dropout(self.head(flattened)).transpose(1, 2)
        if cfg.normalize:
            prediction = prediction*scale+mean
        self.last_selected_indices, self.last_routing = indices, routing
        return prediction
