import torch

from mixhub.model.tensor_types import *

_LARGE_NUM = torch.finfo(torch.float32).max
_LARGENEG_NUM = torch.finfo(torch.float32).min


def masked_sum(x: MixTensor, mask: MaskTensor) -> EmbTensor:
    """Masked sum of a 3D tensor along the second dimension."""
    x_masked = torch.einsum("ijk, ij -> ijk", x, mask)
    return torch.einsum("ijk->ik", x_masked)


def masked_mean(x: MixTensor, mask: MaskTensor) -> EmbTensor:
    """Masked mean of a 3D tensor along the second dimension."""
    x_masked = torch.einsum("ijk, ij -> ijk", x, mask)
    counts = mask.sum(dim=1, keepdim=True)
    mean = torch.einsum("ijk->ik", x_masked) / counts
    return mean


def masked_variance(x: MixTensor, mask: MaskTensor) -> EmbTensor:
    """Masked variance of a 3D tensor along the second dimension."""
    # Assumes ddf=0
    x_masked = torch.einsum("ijk, ij -> ijk", x, mask)
    counts = mask.sum(dim=1, keepdim=True)
    mean = torch.einsum("ijk->ik", x_masked) / counts
    squared_diff = (x - mean.unsqueeze(1)) ** 2
    diff_masked = torch.einsum("ijk, ij -> ijk", squared_diff, mask)
    var = torch.einsum("ijk->ik", diff_masked) / counts
    return var


def masked_min(x: MixTensor, mask: MaskTensor) -> EmbTensor:
    """Masked min of a 3D tensor along the second dimension."""
    x_masked = torch.where(mask.unsqueeze(-1), x, _LARGE_NUM)
    min_values, _ = torch.min(x_masked, dim=1)
    return min_values


def masked_max(x: MixTensor, mask: MaskTensor) -> EmbTensor:
    """Masked max of a 3D tensor along the second dimension."""
    x_masked = torch.where(mask.unsqueeze(-1), x, _LARGENEG_NUM)
    min_values, _ = torch.max(x_masked, dim=1)
    return min_values


if __name__ == "__main__":
    import numpy as np

    # [B, M, D] -> [B, D]
    x = np.array([[1, 2, 3], [4, 5, 4]]).astype(np.float32)
    mask = np.array([[1, 1, 0], [1, 0, 1]]).astype(bool)
    expected_mean = np.array([[1.5], [4.0]])
    expected_var = np.array([[0.25], [0.0]])
    expected_min = np.array([[1.0], [4.0]])
    expected_max = np.array([[2.0], [4.0]])
    print("== Original data")
    print(x.shape, mask.shape)
    print(x)
    print("Mean", expected_mean)
    print("Var", expected_var)
    print("Min", expected_min)
    print("== Repeated data")
    n_repeat = 4
    expected_mean = np.repeat(expected_mean, n_repeat, axis=1)
    expected_var = np.repeat(expected_var, n_repeat, axis=1)
    expected_min = np.repeat(expected_min, n_repeat, axis=1)
    expected_max = np.repeat(expected_max, n_repeat, axis=1)
    print(
        expected_mean.shape, expected_var.shape, expected_min.shape, expected_max.shape
    )
    x = np.repeat(x[:, :, np.newaxis], n_repeat, axis=2)
    actual_mean = masked_mean(torch.from_numpy(x), torch.from_numpy(mask)).numpy()
    print(expected_mean.shape)
    print(actual_mean.shape)
    np.testing.assert_allclose(actual_mean, expected_mean)
    actual_var = masked_variance(torch.from_numpy(x), torch.from_numpy(mask)).numpy()
    print(expected_var.shape)
    print(actual_var.shape)
    np.testing.assert_allclose(actual_var, expected_var)
    actual_min = masked_min(torch.from_numpy(x), torch.from_numpy(mask)).numpy()
    print(expected_min.shape)
    print(actual_min.shape)
    np.testing.assert_allclose(actual_min, expected_min)
    actual_max = masked_max(torch.from_numpy(x), torch.from_numpy(mask)).numpy()
    print(expected_max.shape)
    print(actual_max.shape)
    np.testing.assert_allclose(actual_max, expected_max)
