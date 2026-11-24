"""Check cached execution against the same numerical model and fixed metadata."""
from contextlib import ExitStack
import random

import numpy as np
import torch

from .execution import install
from .model import Spare
from .offline import OfflineSpare, ReplayPolicy


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def check_cached_execution(cfg, cache, loader, execution):
    device = loader.values.device
    ids = torch.tensor([0], device=device)
    indices, route = cache.lookup("train", ids)
    batch = loader.batch(ids)
    seed(cfg.seed)
    reference = Spare(cfg, ReplayPolicy(indices, route)).float().to(device)
    cached = OfflineSpare(cfg).float().to(device)
    cached.load_state_dict(reference.state_dict())
    prediction_max = gradient_max = input_max = 0.
    with ExitStack() as stack:
        stack.enter_context(install(reference, execution))
        stack.enter_context(install(cached, execution))
        for training in (False, True):
            reference.train(training)
            cached.train(training)
            reference.zero_grad(set_to_none=True)
            cached.zero_grad(set_to_none=True)
            x = batch[0].detach().clone().requires_grad_()
            y = batch[0].detach().clone().requires_grad_()
            seed(43)
            expected = reference(x, batch[2])
            expected.square().mean().backward()
            seed(43)
            actual = cached(y, batch[2], metadata=(indices, route))
            actual.square().mean().backward()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(y.grad, x.grad, rtol=1e-5, atol=1e-7)
            prediction_max = max(prediction_max, (actual-expected).abs().max().item())
            input_max = max(input_max, (y.grad-x.grad).abs().max().item())
            for (name, a), (other, b) in zip(reference.named_parameters(), cached.named_parameters()):
                if name != other:
                    raise AssertionError("Parameter names changed")
                if a.grad is None:
                    if b.grad is not None:
                        raise AssertionError(f"Unexpected gradient: {name}")
                else:
                    torch.testing.assert_close(b.grad, a.grad, rtol=1e-5, atol=1e-7)
                    gradient_max = max(gradient_max, (b.grad-a.grad).abs().max().item())
    return dict(status="passed", train_and_eval=True, max_prediction_abs=prediction_max,
                max_input_gradient_abs=input_max, max_parameter_gradient_abs=gradient_max,
                gradient_rtol=1e-5, gradient_atol=1e-7,
                scope="Same weights and fixed metadata; not epoch-wise resampling equivalence")
