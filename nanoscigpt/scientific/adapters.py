"""Small, explicit representation adapters used by the classroom fixtures."""

import torch


def patchify_2d(values, patch_size):
    """Convert ``(batch, channels, height, width)`` arrays into spatial tokens."""
    if values.ndim != 4:
        raise ValueError(f"patchify_2d expects 4 dimensions, got {tuple(values.shape)}")
    batch, channels, height, width = values.shape
    if height % patch_size or width % patch_size:
        raise ValueError("height and width must be divisible by patch_size")
    patches = values.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    return patches.view(batch, -1, channels * patch_size * patch_size)


def patchify_1d(values, patch_size):
    """Convert ``(batch, channels, length)`` arrays into ordered 1D tokens."""
    if values.ndim != 3:
        raise ValueError(f"patchify_1d expects 3 dimensions, got {tuple(values.shape)}")
    batch, channels, length = values.shape
    if length % patch_size:
        raise ValueError("length must be divisible by patch_size")
    patches = values.unfold(2, patch_size, patch_size)
    return patches.permute(0, 2, 1, 3).contiguous().view(
        batch, length // patch_size, channels * patch_size
    )


def pairwise_distance_tokens(points):
    """Represent a point set by its translation/rotation-invariant distance rows."""
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"expected (batch, points, 3), got {tuple(points.shape)}")
    return torch.cdist(points, points)


def periodic_distances(fractional, lattice):
    """Pair distances under the minimum-image convention for periodic cells."""
    if fractional.ndim != 3 or fractional.shape[-1] != 3:
        raise ValueError("fractional coordinates must have shape (batch, nodes, 3)")
    if lattice.ndim != 3 or lattice.shape[-2:] != (3, 3):
        raise ValueError("lattice must have shape (batch, 3, 3)")
    displacement = fractional[:, :, None, :] - fractional[:, None, :, :]
    displacement = displacement - torch.round(displacement)
    cartesian = torch.einsum("bijd,bdk->bijk", displacement, lattice)
    return torch.linalg.vector_norm(cartesian, dim=-1)
