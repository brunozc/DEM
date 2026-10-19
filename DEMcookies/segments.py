"""Closest-point queries between points and line segments."""

from typing import Tuple

import numpy as np
import numpy.typing as npt


def project_onto_segments(points: npt.NDArray[np.float64], starts: npt.NDArray[np.float64],
                          ends: npt.NDArray[np.float64]) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Find the point of every segment closest to every point.

    Args:
        points: points to project, shape (n, 2) [m].
        starts: segment start points, shape (s, 2) [m].
        ends: segment end points, shape (s, 2) [m].

    Returns:
        The segment parameters t in [0, 1] of the closest points, shape (n, s), and the vectors from the closest
        points to the points, shape (n, s, 2) [m].
    """
    direction = ends - starts
    length_sq = np.einsum("ij,ij->i", direction, direction)
    relative = points[:, None, :] - starts[None, :, :]
    t = np.clip(np.einsum("nsk,sk->ns", relative, direction) / length_sq, 0.0, 1.0)
    offset = relative - t[:, :, None] * direction[None, :, :]
    return t, offset
