"""Vector operations that hold in two and in three dimensions.

In a plane every rotation is about the out-of-plane axis, so in 2D an angular velocity, a torque or a rotation is a
scalar, the out-of-plane component of the vector it is in 3D. These helpers apply one formula to both cases. In 2D
they evaluate exactly the expressions of the planar model, so that its results do not change by a single bit.
"""

from types import ModuleType
from typing import Tuple

import numpy as np
import numpy.typing as npt

from .backend import array_module


def norm(vectors: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Length of each vector along the last axis.

    Args:
        vectors: vectors with 2 or 3 components along the last axis.

    Returns:
        The lengths, with the last axis removed.
    """
    if vectors.shape[-1] == 2:
        return np.asarray(np.hypot(vectors[..., 0], vectors[..., 1]))
    return np.asarray(np.sqrt(np.einsum("...k,...k->...", vectors, vectors)))


def cross(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Cross product of each pair of vectors; in 2D only its out-of-plane component, a scalar.

    Args:
        a: first vectors, shape (k, 2) or (k, 3).
        b: second vectors, of the same shape.

    Returns:
        The cross products a x b, shape (k,) in 2D and (k, 3) in 3D.
    """
    if a.shape[-1] == 2:
        return np.asarray(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
    return np.asarray(np.cross(a, b))


def cross_3d(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Cross product of 3D vectors along the last axis, for arrays of any leading shapes that broadcast.

    It spells the components out, which is much faster than ``numpy.cross`` on the small arrays of a contact batch.

    Args:
        a: first vectors, NumPy or CuPy, shape (..., 3).
        b: second vectors, shape (..., 3).

    Returns:
        The cross products a x b, of the broadcast shape, on the device of the vectors.
    """
    out = array_module(a, b).empty(np.broadcast_shapes(a.shape, b.shape))
    out[..., 0] = a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1]
    out[..., 1] = a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2]
    out[..., 2] = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    return out


def spin_cross(spin: npt.NDArray[np.float64], vectors: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Cross product of rotation vectors with vectors, e.g. the velocity omega x r of a point of a spinning body.

    Args:
        spin: rotation vectors, shape (k,) in 2D, where they point out of the plane, or (k, 3) in 3D.
        vectors: vectors, shape (k, 2) or (k, 3).

    Returns:
        The products spin x vectors, of the shape of the vectors.
    """
    if spin.ndim == 1:
        return np.asarray(spin[:, None] * np.column_stack((-vectors[:, 1], vectors[:, 0])))
    return np.asarray(np.cross(spin, vectors))


def scale(factors: npt.NDArray[np.float64], values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Multiply each scalar or vector by its own factor.

    Args:
        factors: one factor per value, shape (k,).
        values: scalars, shape (k,), or vectors, shape (k, d).

    Returns:
        The scaled values, of the shape of the values.
    """
    if values.ndim == 1:
        return np.asarray(factors * values)
    return np.asarray(factors[:, None] * values)


def scatter_add(index: npt.NDArray[np.int64], values: npt.NDArray[np.float64], size: int) -> npt.NDArray[np.float64]:
    """Sum scalars or vectors into the bins given by an index, e.g. contact forces onto the particles.

    The sums are the same from run to run on either device. On a GPU, where ``bincount`` adds the values with atomic
    operations in an order that changes from run to run, the values are sorted by bin and summed bin by bin instead.

    Args:
        index: bin of each value, shape (k,).
        values: scalars, shape (k,), or vectors, shape (k, d), NumPy or CuPy.
        size: number of bins.

    Returns:
        The sum in each bin, shape (size,) or (size, d), on the device of the values.
    """
    xp = array_module(values)
    if xp is not np:
        return _sorted_sum(xp, index, values, size)
    if values.ndim == 1:
        return np.asarray(np.bincount(index, values, size), dtype=np.float64)
    return np.column_stack([np.bincount(index, values[:, k], size) for k in range(values.shape[1])]).reshape(size, -1)


def _sorted_sum(xp: ModuleType, index: npt.NDArray[np.int64], values: npt.NDArray[np.float64],
                size: int) -> npt.NDArray[np.float64]:
    """Sum values into bins in an order that never changes: sorted by bin, and by position within a bin.

    The values of each bin are laid out in a row of a table padded with zeros, and the rows are summed.

    Args:
        xp: array module of the values.
        index: bin of each value, shape (k,).
        values: scalars, shape (k,), or vectors, shape (k, d).
        size: number of bins.

    Returns:
        The sum in each bin, shape (size,) or (size, d).
    """
    count = index.size
    if count == 0:
        return xp.zeros((size, ) + values.shape[1:])
    # a key unique to every value makes the order independent of how the sort treats ties
    order = xp.argsort(index.astype(np.int64) * count + xp.arange(count))
    sorted_index = index[order]
    # integer counts are exact whatever the order of the atomic additions
    counts = xp.bincount(index, minlength=size)
    rank = xp.arange(count) - (xp.cumsum(counts) - counts)[sorted_index]
    table = xp.zeros((size, int(counts.max())) + values.shape[1:])
    table[sorted_index, rank] = values[order]
    return xp.asarray(table.sum(axis=1))


def rotation_shape(count: int, dimension: int) -> Tuple[int, ...]:
    """Shape of an array of rotations, angular velocities or torques.

    Args:
        count: number of rotations.
        dimension: number of space dimensions, 2 or 3.

    Returns:
        (count,) in 2D, where rotations are scalars, and (count, 3) in 3D.
    """
    return (count, ) if dimension == 2 else (count, 3)
