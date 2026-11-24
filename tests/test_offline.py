"""The cached path must preserve the specified fixed selections, outputs and gradients."""
import unittest
from unittest.mock import patch

import torch

from spare.offline import MetadataCache, OfflineSpare, ReplayPolicy, normalized
from spare.model import Config, Spare
from spare.policy import ReferencePolicy, symbolize


class ForbiddenPolicy:
    def select(self, *args):
        raise AssertionError("Online selection inside offline training")

    def route(self, *args):
        raise AssertionError("Online grouping inside offline training")


class OfflineTests(unittest.TestCase):
    def test_outputs_gradients_and_cache_order(self):
        torch.set_num_threads(1)
        for retention in (.1, .5, 1.):
            torch.manual_seed(2021)
            cfg = Config(length=64, horizon=8, variables=3, width=16,
                         feedforward=32, heads=2, layers=2, retention=retention)
            raw = torch.randn(4, 64, 3)
            x, _, _ = normalized(raw, True)
            patches = x.transpose(1, 2).unfold(-1, 16, 16)
            cuts = Spare(cfg).breakpoints
            words = symbolize(patches, 4, cuts)
            policy = ReferencePolicy()
            indices = policy.select(words, retention, torch.Generator().manual_seed(3))
            selected = words.gather(2, indices[..., None].expand(*indices.shape, 4))
            route = policy.route(selected.reshape(12, indices.shape[-1], 4), 4)
            cache = MetadataCache(cfg)
            cache.allocate("train", 4, "cpu")
            cache.store("train", torch.arange(4), indices, route)
            order = torch.tensor([3, 0, 2])
            cached_indices, cached_route = cache.lookup("train", order)
            torch.testing.assert_close(cached_indices, indices[order], rtol=0, atol=0)
            reference = Spare(cfg, ReplayPolicy(cached_indices, cached_route), "math")
            cached = OfflineSpare(cfg, ForbiddenPolicy(), "math")
            cached.load_state_dict(reference.state_dict())
            self.assertEqual(tuple(reference.state_dict()), tuple(cached.state_dict()))
            for training in (False, True):
                reference.train(training)
                cached.train(training)
                inputs = [raw[order].clone().requires_grad_() for _ in range(2)]
                reference.zero_grad(set_to_none=True)
                cached.zero_grad(set_to_none=True)
                torch.manual_seed(43)
                expected = reference(inputs[0])
                expected.square().mean().backward()
                torch.manual_seed(43)
                with patch("spare.model.symbolize", side_effect=AssertionError("Online SAX")):
                    actual = cached(inputs[1], metadata=(cached_indices, cached_route))
                actual.square().mean().backward()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(inputs[1].grad, inputs[0].grad, rtol=0, atol=0)
                for a, b in zip(reference.parameters(), cached.parameters()):
                    if a.grad is None:
                        self.assertIsNone(b.grad)
                    else:
                        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
            # A fixed-window cache does not change when batches or epochs are shuffled.
            again, _ = cache.lookup("train", order)
            torch.testing.assert_close(again, cached_indices, rtol=0, atol=0)
            with self.assertRaises(ValueError):
                cache.lookup("train", [-1])
            with self.assertRaises(ValueError):
                cache.lookup("val", [0])


if __name__ == "__main__":
    unittest.main()
