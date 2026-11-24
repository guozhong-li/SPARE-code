"""Paper examples, discrete-policy oracle and independent numerical checks."""
import copy
import math
import unittest

import torch

from spare.policy import ReferencePolicy, addresses, symbolize
from spare.model import Config, Spare, attention


class Tests(unittest.TestCase):
    def test_patch_normalization_preserves_paper_example(self):
        x = torch.tensor([-1., -.25, .25, 1.]).repeat_interleave(4)
        patches = torch.stack((x, 10+2*x, -3+.5*x))[None, None]
        probabilities = torch.linspace(0, 1, 5)[1:-1]
        breaks = math.sqrt(2.)*torch.erfinv(2*probabilities-1)
        words = symbolize(patches, 4, breaks)
        self.assertEqual(words.flatten(0, 2).tolist(), [[0, 1, 2, 3]]*3)

    def test_cbf_collisions_and_running_grouping(self):
        self.assertEqual(addresses((0, 1, 2, 3)), [157, 248, 205])
        self.assertEqual(addresses((2, 3, 0, 1)), [221, 24, 205])
        words = torch.tensor([[[3, 2, 0, 1], [0, 1, 2, 3], [0, 1, 3, 2], [2, 3, 0, 1]]])
        route = ReferencePolicy().route(words, 2)
        self.assertTrue(torch.all(route["counts"] <= route["capacity"]))
        self.assertEqual(route["counts"].sum().item(), 4)
        self.assertEqual(len(set(route["slots"][0].tolist())), 4)

    def test_retention_latest_representatives_and_full_bypass(self):
        policy = ReferencePolicy()
        words = torch.zeros(2, 3, 32, 4, dtype=torch.long)
        generator = torch.Generator().manual_seed(102024)
        before = generator.get_state()
        full = policy.select(words, 1., generator)
        self.assertTrue(torch.equal(full, torch.arange(32).expand(2, 3, 32)))
        self.assertTrue(torch.equal(before, generator.get_state()))
        half = policy.select(words, .5, generator)
        self.assertTrue(torch.all(half[..., -1] == 31))
        self.assertEqual(half.shape, (2, 3, 16))
        tenth = policy.select(words, .1, generator)
        self.assertEqual(tenth.shape, (2, 3, 3))
        self.assertTrue(torch.all(tenth[..., 1:] > tenth[..., :-1]))

    def test_sdpa_matches_independent_math_and_gradients_without_dropout(self):
        for n in (1, 3, 16):
            source = [torch.randn(2, 2, n, 4) for _ in range(3)]
            outputs = []
            for backend in ("math", "sdpa"):
                tensors = [x.clone().requires_grad_() for x in source]
                out = attention(*tensors, 0., backend=backend)
                out.square().mean().backward()
                outputs.append((out, [x.grad for x in tensors]))
            torch.testing.assert_close(outputs[0][0], outputs[1][0], atol=1e-5, rtol=1e-5)
            for a, b in zip(outputs[0][1], outputs[1][1]):
                torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)

    def test_complete_model_backends_and_training_smoke(self):
        for ratio in (.1, .5, 1.):
            cfg = Config(length=128, horizon=8, variables=3, width=16, feedforward=32,
                         heads=4, layers=2, retention=ratio, dropout=0.)
            math_model = Spare(cfg, attention_backend="math")
            fast_model = Spare(cfg)
            fast_model.load_state_dict(math_model.state_dict())
            x, marks = torch.randn(2, 128, 3), torch.randn(2, 128, 4)
            expected, actual = math_model(x, marks), fast_model(x, marks)
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
            self.assertTrue(torch.equal(math_model.last_selected_indices, fast_model.last_selected_indices))
            for name in ("labels", "slots", "counts", "medoids"):
                self.assertTrue(torch.equal(math_model.last_routing[name], fast_model.last_routing[name]))
            expected.square().mean().backward()
            actual.square().mean().backward()
            for a, b in zip(math_model.parameters(), fast_model.parameters()):
                torch.testing.assert_close(a.grad, b.grad, atol=1e-5, rtol=1e-4)
        cfg = Config(length=32, horizon=8, variables=3, width=16, feedforward=32,
                     heads=4, layers=1, dropout=.1)
        net = Spare(cfg).train()
        optimizer = torch.optim.Adam(net.parameters(), lr=.0005)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss = net(torch.randn(2, 32, 3), torch.randn(2, 32, 4)).square().mean()
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters()))
            optimizer.step()


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main()
