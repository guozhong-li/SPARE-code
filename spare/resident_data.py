"""Exact benchmark windows gathered from GPU-resident scaled arrays."""
import numpy as np
import torch


class ResidentWindows:
    def __init__(self, dataset, batch_size, length, label, horizon, shuffle, seed, device="cuda"):
        self.dataset, self.batch_size, self.shuffle = dataset, batch_size, shuffle
        self.generator = torch.Generator().manual_seed(seed)
        self.values = torch.as_tensor(np.ascontiguousarray(dataset.data_x, dtype=np.float32), device=device)
        same = (dataset.data_x.shape == dataset.data_y.shape
                and dataset.data_x.strides == dataset.data_y.strides
                and dataset.data_x.__array_interface__["data"][0] == dataset.data_y.__array_interface__["data"][0])
        self.targets = self.values if same else torch.as_tensor(
            np.ascontiguousarray(dataset.data_y, dtype=np.float32), device=device)
        self.marks = torch.as_tensor(np.ascontiguousarray(dataset.data_stamp, dtype=np.float32), device=device)
        self.input_offsets = torch.arange(length, device=device)
        self.target_offsets = torch.arange(length-label, length+horizon, device=device)

    def __len__(self):
        return (len(self.dataset)+self.batch_size-1)//self.batch_size

    def batch(self, indices):
        x = indices[:, None]+self.input_offsets[None]
        y = indices[:, None]+self.target_offsets[None]
        return self.values[x], self.targets[y], self.marks[x], self.marks[y]

    def __iter__(self):
        count = len(self.dataset)
        order = torch.randperm(count, generator=self.generator) if self.shuffle else torch.arange(count)
        order = order.to(self.values.device)
        for start in range(0, count, self.batch_size):
            yield self.batch(order[start:start+self.batch_size])


def verify_windows(loader):
    indices = sorted({0, min(17, len(loader.dataset)-1), len(loader.dataset)-1})
    actual = loader.batch(torch.tensor(indices, device=loader.values.device))
    for field, tensor in enumerate(actual):
        expected = torch.from_numpy(np.stack([loader.dataset[i][field] for i in indices])).float().to(tensor.device)
        torch.testing.assert_close(tensor, expected, rtol=0, atol=0)
    return dict(status="passed", indices=indices, exact_input_target_and_mark_values=True)
