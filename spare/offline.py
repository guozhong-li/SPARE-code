"""Window-fixed discrete metadata; the numerical model and weights are unchanged."""
import math
import torch

from .model import Spare, plan
from .policy import symbolize


def normalized(x, enabled):
    if not enabled:
        return x, None, None
    mean = x.mean(1, keepdim=True).detach()
    centered = x-mean
    scale = torch.sqrt(centered.var(1, keepdim=True, unbiased=False)+1e-5)
    return centered/scale, mean, scale


class MetadataCache:
    """Compact GPU-resident indices keyed by split and original window ID."""
    def __init__(self, cfg):
        self.cfg, self.splits = cfg, {}
        self.kept = max(1, round(cfg.length//cfg.patch*cfg.retention))
        self.groups = min(cfg.groups, self.kept)
        self.capacity = math.ceil(self.kept/self.groups)
        if max(cfg.length//cfg.patch, self.groups*self.capacity) >= 32768:
            raise ValueError("Metadata does not fit int16")

    def allocate(self, split, windows, device):
        if split in self.splits:
            raise ValueError("A split cache may be constructed only once")
        self.splits[split] = {
            key: torch.empty(windows, self.cfg.variables, length, dtype=torch.int16, device=device)
            for key, length in (("indices", self.kept), ("labels", self.kept),
                                ("slots", self.kept), ("counts", self.groups))}

    def store(self, split, ids, indices, route):
        values = dict(indices=indices, **{k: route[k] for k in ("labels", "slots", "counts")})
        for key, value in values.items():
            target = self.splits[split][key]
            target[ids] = value.reshape(len(ids), self.cfg.variables, -1).to(torch.int16)

    def lookup(self, split, ids, validate=True):
        if split not in self.splits:
            raise ValueError("Split cache is unavailable")
        arrays = self.splits[split]
        ids = torch.as_tensor(ids, dtype=torch.long, device=arrays["indices"].device)
        if ids.ndim != 1 or (validate and ((ids < 0).any() or (ids >= len(arrays["indices"])).any())):
            raise ValueError("Invalid window ID")
        selected = {key: value[ids].long() for key, value in arrays.items()}
        indices = selected.pop("indices")
        route = {key: value.reshape(-1, value.shape[-1]) for key, value in selected.items()}
        route.update(groups=self.groups, capacity=self.capacity, input_tokens=self.kept)
        return indices, route

    def bytes(self):
        return sum(x.numel()*x.element_size() for split in self.splits.values() for x in split.values())


@torch.no_grad()
def build_cache(cfg, loaders, policy, chunk):
    cache = MetadataCache(cfg)
    for offset, (split, loader) in enumerate(loaders.items()):
        device = loader.values.device
        cache.allocate(split, len(loader.dataset), device)
        generator = torch.Generator(device=device).manual_seed(cfg.seed+100003+offset)
        probs = torch.linspace(0, 1, cfg.alphabet+1)[1:-1]
        cuts = (math.sqrt(2.)*torch.erfinv(2*probs-1)).to(device)
        for start in range(0, len(loader.dataset), chunk):
            ids = torch.arange(start, min(start+chunk, len(loader.dataset)), device=device)
            x = loader.batch(ids)[0].float()
            x, _, _ = normalized(x, cfg.normalize)
            patches = x.transpose(1, 2).unfold(-1, cfg.patch, cfg.patch)
            words = symbolize(patches, cfg.segment, cuts)
            indices = policy.select(words, cfg.retention, generator)
            chosen = words.gather(2, indices[..., None].expand(*indices.shape, words.shape[-1]))
            route = policy.route(chosen.reshape(-1, cache.kept, words.shape[-1]), cfg.groups)
            cache.store(split, ids, indices, route)
    return cache


class ReplayPolicy:
    def __init__(self, indices, route):
        self.indices, self.routing = indices, route

    def select(self, *args, **kwargs):
        return self.indices

    def route(self, *args, **kwargs):
        return self.routing


class OfflineSpare(Spare):
    def forward(self, x, marks=None, decoder=None, future_marks=None, metadata=None):
        if metadata is None:
            raise ValueError("Offline forward requires cached window metadata")
        cfg = self.cfg
        if x.shape[1:] != (cfg.length, cfg.variables):
            raise ValueError("Input shape disagrees with model configuration")
        x, mean, scale = normalized(x, cfg.normalize)
        indices, routing = metadata
        patches = x.transpose(1, 2).unfold(-1, cfg.patch, cfg.patch)
        raw = patches.gather(2, indices[..., None].expand(*indices.shape, cfg.patch))
        batch, variables, kept, _ = raw.shape
        route = plan(routing)
        numerical = self.patch_embedding(raw.reshape(batch*variables, kept, cfg.patch))
        numerical = numerical+self.position[0, indices.reshape(batch*variables, kept)]
        token = self.global_token.expand(batch, -1, -1, -1).reshape(batch*variables, 1, cfg.width)
        numerical = self.embedding_dropout(torch.cat((numerical, token), 1))
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
