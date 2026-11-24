"""Fresh training with separate offline preprocessing, validation and final test."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from .data import load_datasets
from .execution import install
from .model import Config, Spare
from .offline import OfflineSpare, build_cache
from .policy import FusedPolicy, ReferencePolicy
from .resident_data import ResidentWindows, verify_windows
from .verification import check_cached_execution, seed

ROOT = Path(__file__).resolve().parents[1]
PRESETS = ROOT / "configs" / "benchmarks.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def loaders(preset, opts, device):
    datasets = load_datasets(preset, opts.data_root, opts.length, opts.horizon,
                             min(preset["label_len"], opts.length))
    result = {}
    for offset, (split, dataset) in enumerate(datasets.items()):
        result[split] = ResidentWindows(dataset, opts.batch_size, opts.length,
                                       min(preset["label_len"], opts.length), opts.horizon,
                                       split == "train", opts.seed + 17 + offset, device=device)
        verify_windows(result[split])
    return result


def batches(loader, limit=None):
    n = len(loader.dataset)
    order = torch.randperm(n, generator=loader.generator) if loader.shuffle else torch.arange(n)
    order = order.to(loader.values.device)
    for step, start in enumerate(range(0, n, loader.batch_size)):
        if limit is not None and step >= limit:
            break
        ids = order[start:start + loader.batch_size]
        yield ids, loader.batch(ids)


def predict(net, batch, horizon, metadata):
    x, y, marks, _ = batch
    if metadata is None:
        prediction = net(x, marks)
    else:
        prediction = net(x, marks, metadata=metadata)
    prediction, target = prediction[:, -horizon:], y[:, -horizon:]
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target shapes differ")
    return prediction, target


def evaluate(net, loader, horizon, cache, split, limit):
    net.eval()
    net.reset_policy_rng()
    totals = torch.zeros(2, dtype=torch.float64, device=loader.values.device)
    elements = windows = 0
    with torch.no_grad():
        for ids, batch in batches(loader, limit):
            metadata = cache.lookup(split, ids, validate=False) if cache is not None else None
            prediction, target = predict(net, batch, horizon, metadata)
            error = (prediction-target).double()
            totals += torch.stack((error.square().sum(), error.abs().sum()))
            elements += error.numel()
            windows += len(prediction)
    mse, mae = (totals/elements).tolist()
    if not all(np.isfinite(x) for x in (mse, mae)):
        raise ValueError("Nonfinite evaluation")
    if limit is None and windows != len(loader.dataset):
        raise ValueError("Evaluation split was truncated")
    return dict(mse=mse, mae=mae, windows=windows, elements=elements)


def save_checkpoint(net, path):
    temporary = path.with_suffix(".pth.tmp")
    torch.save({key: value.detach().cpu() for key, value in net.state_dict().items()}, temporary)
    temporary.replace(path)


def parse_args(argv=None):
    presets = json.loads(PRESETS.read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(presets), default="Weather")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="New run directory; existing directories are never overwritten")
    parser.add_argument("--mode", choices=("offline", "online"), default="offline")
    parser.add_argument("--backend", choices=("reference", "fused"), default="reference")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--execution", choices=("native", "all_math_singleton"),
                        default="all_math_singleton")
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--retention", type=float, choices=(.1, .5, 1.), default=.5)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--preprocess-batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--d-ff", type=int, default=None)
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--verify-data", action="store_true",
                        help="Require the raw CSV hash from the original local campaign")
    parser.add_argument("--smoke", action="store_true",
                        help="Two epochs/two batches per split; not a benchmark result")
    opts = parser.parse_args(argv)
    preset = presets[opts.dataset]
    for key, source in (("batch_size", "batch_size"), ("epochs", "train_epochs"),
                        ("patience", "patience"), ("d_model", "d_model"),
                        ("d_ff", "d_ff"), ("heads", "n_heads"), ("layers", "e_layers"),
                        ("learning_rate", "learning_rate")):
        if getattr(opts, key) is None:
            setattr(opts, key, preset[source])
    if min(opts.batch_size, opts.epochs, opts.patience, opts.preprocess_batch_size) < 1:
        parser.error("Batch sizes, epochs and patience must be positive")
    if not np.isfinite(opts.learning_rate) or opts.learning_rate <= 0:
        parser.error("Learning rate must be finite and positive")
    if opts.backend == "reference" and opts.device != "cpu":
        parser.error("The readable reference selector supports CPU only; use fused for CUDA")
    if opts.backend == "fused" and opts.device != "cuda":
        parser.error("Fused kernels require a CUDA device")
    if opts.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    return opts, preset


def main(argv=None):
    opts, preset = parse_args(argv)
    cfg = Config(length=opts.length, horizon=opts.horizon, variables=preset["enc_in"],
                 width=opts.d_model, feedforward=opts.d_ff, heads=opts.heads, layers=opts.layers,
                 patch=preset["patch_len"], segment=preset["sax_seg_len"], alphabet=preset["sax_alphabet"],
                 retention=opts.retention, groups=preset["num_groups"], dropout=preset["dropout"], seed=opts.seed)
    device = torch.device(opts.device)
    data_path = opts.data_root / preset["data_path"]
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset CSV not found: {data_path}")
    data_hash = sha(data_path)
    if opts.verify_data:
        expected = json.loads((ROOT / "configs" / "dataset_hashes.json").read_text())[opts.dataset]
        if data_hash != expected:
            raise ValueError("Dataset hash differs from the original local campaign")
    opts.output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("TRITON_CACHE_DIR", str(opts.output.resolve() / "triton-cache"))
    torch.set_num_threads(4)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    seed(opts.seed)
    result = dict(status="initializing", smoke=opts.smoke, fresh_training=True,
                  started_utc=datetime.now(timezone.utc).isoformat(), model=asdict(cfg),
                  options={k: str(v) if isinstance(v, Path) else v for k, v in vars(opts).items()},
                  torch=torch.__version__, device=str(device), gpu=torch.cuda.get_device_name(device)
                  if device.type == "cuda" else None, csv_sha256=data_hash,
                  source_sha256={str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / "spare").rglob("*.py"))},
                  preprocessing_seconds=0., training_seconds=None, test=None, history=[],
                  timing="Fresh model/loader/Adam construction through early stop or epoch cap; includes validation, checkpoints and first-use numerical compilation. Offline preprocessing, correctness gates and final test are separate.")
    cache = data = None
    output = opts.output / "result.json"
    write_json(output, result)
    try:
        if opts.mode == "offline":
            sync(device)
            start = time.perf_counter()
            data = loaders(preset, opts, device)
            policy = FusedPolicy(device=device) if opts.backend == "fused" else ReferencePolicy()
            cache = build_cache(cfg, data, policy, opts.preprocess_batch_size)
            sync(device)
            result.update(preprocessing_seconds=time.perf_counter()-start, cache_bytes=cache.bytes(),
                          selection_mode="One fixed draw per split/window; reuse metadata across epochs")
            start = time.perf_counter()
            result["cache_gate"] = check_cached_execution(cfg, cache, data["train"], opts.execution)
            sync(device)
            result["verification_seconds_excluded"] = time.perf_counter()-start
            del data, policy
            data = None
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        seed(opts.seed)
        sync(device)
        started = time.perf_counter()
        if cache is not None:
            net = OfflineSpare(cfg).float().to(device)
        else:
            policy = FusedPolicy(device=device) if opts.backend == "fused" else ReferencePolicy()
            net = Spare(cfg, policy).float().to(device)
            result["selection_mode"] = "Online selection and grouping; diversity draws follow visits"
        data = loaders(preset, opts, device)
        optimizer = torch.optim.Adam(net.parameters(), lr=opts.learning_rate)
        result.update(status="training", parameters=sum(p.numel() for p in net.parameters()),
                      windows={split: len(loader.dataset) for split, loader in data.items()})
        best, stale = float("inf"), 0
        limit = 2 if opts.smoke else None
        with install(net, opts.execution):
            for epoch in range(2 if opts.smoke else opts.epochs):
                net.train()
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                sync(device)
                begin = time.perf_counter()
                loss_sum = 0.
                elements = windows = 0
                for ids, batch in batches(data["train"], limit):
                    metadata = cache.lookup("train", ids, validate=False) if cache is not None else None
                    optimizer.zero_grad(set_to_none=True)
                    prediction, target = predict(net, batch, opts.horizon, metadata)
                    loss = (prediction-target).square().mean()
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite training loss")
                    loss.backward()
                    optimizer.step()
                    loss_sum += loss.detach().item()*target.numel()
                    elements += target.numel()
                    windows += len(prediction)
                    del prediction, target, loss
                sync(device)
                if limit is None and windows != len(data["train"].dataset):
                    raise ValueError("Training split was truncated")
                entry = dict(epoch=epoch+1, train_mse=loss_sum/elements, train_windows=windows,
                             train_seconds=time.perf_counter()-begin,
                             peak_allocated_mib=torch.cuda.max_memory_allocated(device)/2**20
                             if device.type == "cuda" else None)
                begin = time.perf_counter()
                entry["validation"] = evaluate(net, data["val"], opts.horizon, cache, "val", limit)
                sync(device)
                entry["validation_seconds"] = time.perf_counter()-begin
                if entry["validation"]["mse"] < best:
                    best, stale = entry["validation"]["mse"], 0
                    save_checkpoint(net, opts.output / "best.pth")
                    result["best_epoch"] = epoch+1
                else:
                    stale += 1
                result["history"].append(entry)
                result.update(best_validation_mse=best, epochs_completed=epoch+1)
                write_json(output, result)
                print(json.dumps(entry), flush=True)
                if stale >= opts.patience:
                    break
                for group in optimizer.param_groups:
                    group["lr"] = opts.learning_rate * .5**epoch
            sync(device)
            result.update(training_seconds=time.perf_counter()-started,
                          stop_reason="validation_early_stopping" if stale >= opts.patience else "maximum_epochs",
                          status="testing", checkpoint_sha256=sha(opts.output / "best.pth"))
            result["preprocessing_plus_training_seconds"] = result["preprocessing_seconds"] + result["training_seconds"]
            write_json(output, result)
            del optimizer
            net.zero_grad(set_to_none=True)
            net.load_state_dict(torch.load(opts.output / "best.pth", map_location=device, weights_only=True))
            begin = time.perf_counter()
            result["test"] = evaluate(net, data["test"], opts.horizon, cache, "test", limit)
            sync(device)
            result.update(final_test_seconds_excluded=time.perf_counter()-begin,
                          status="smoke_complete" if opts.smoke else "complete")
    except Exception as error:
        result.update(status="failed", error=repr(error), training_seconds=None)
        raise
    finally:
        write_json(output, result)
    print(json.dumps({key: result[key] for key in ("status", "preprocessing_seconds", "training_seconds", "test")}),
          flush=True)
