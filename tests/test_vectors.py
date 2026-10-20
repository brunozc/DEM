"""Tests of the vector operations shared by the 2D and 3D models."""

import numpy as np
from numpy.testing import assert_allclose

from DEMcookies.vectors import cross, norm, rotation_shape, scale, scatter_add, spin_cross


class TestVectors:
    """Tests of the helpers that apply one formula in two and in three dimensions."""

    def test_norm(self) -> None:
        """The length must be taken along the last axis, in 2D and in 3D."""
        assert_allclose(norm(np.array([[3.0, 4.0], [0.0, 0.0]])), [5.0, 0.0])
        assert_allclose(norm(np.array([[[1.0, 2.0, 2.0]], [[0.0, 0.0, -3.0]]])), [[3.0], [3.0]])

    def test_cross_is_out_of_plane_component_in_2d(self) -> None:
        """In 2D the cross product must be the scalar z component of the 3D product of the embedded vectors."""
        rng = np.random.default_rng(0)
        a, b = rng.normal(size=(5, 2)), rng.normal(size=(5, 2))
        embedded = np.cross(np.column_stack((a, np.zeros(5))), np.column_stack((b, np.zeros(5))))
        assert_allclose(cross(a, b), embedded[:, 2], rtol=1e-12)
        assert_allclose(cross(np.column_stack((a, a[:, :1])), np.column_stack((b, b[:, :1]))),
                        np.cross(np.column_stack((a, a[:, :1])), np.column_stack((b, b[:, :1]))))

    def test_spin_cross_matches_embedded_rotation(self) -> None:
        """A 2D spin must act like a rotation vector along z on the embedded vectors."""
        rng = np.random.default_rng(1)
        spin, vectors = rng.normal(size=4), rng.normal(size=(4, 2))
        embedded = np.cross(np.column_stack((np.zeros((4, 2)), spin)), np.column_stack((vectors, np.zeros(4))))
        assert_allclose(spin_cross(spin, vectors), embedded[:, :2], rtol=1e-12, atol=1e-15)
        spin_3d, vectors_3d = rng.normal(size=(4, 3)), rng.normal(size=(4, 3))
        assert_allclose(spin_cross(spin_3d, vectors_3d), np.cross(spin_3d, vectors_3d))

    def test_scale(self) -> None:
        """Each scalar or vector must be multiplied by its own factor."""
        assert_allclose(scale(np.array([2.0, 3.0]), np.array([1.0, -1.0])), [2.0, -3.0])
        assert_allclose(scale(np.array([2.0, 3.0]), np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])),
                        [[2.0, 0.0, 2.0], [0.0, 3.0, 0.0]])

    def test_scatter_add(self) -> None:
        """Scalars and vectors must be summed into their bins, empty bins staying zero."""
        index = np.array([0, 2, 0])
        assert_allclose(scatter_add(index, np.array([1.0, 2.0, 3.0]), 4), [4.0, 0.0, 2.0, 0.0])
        vectors = np.array([[1.0, 0.0, 1.0], [0.0, 2.0, 0.0], [3.0, 0.0, -1.0]])
        assert_allclose(scatter_add(index, vectors, 3), [[4.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        assert scatter_add(np.zeros(0, dtype=np.int64), np.zeros((0, 3)), 2).shape == (2, 3)

    def test_rotation_shape(self) -> None:
        """Rotations must be scalars in 2D and vectors in 3D."""
        assert rotation_shape(5, 2) == (5, )
        assert rotation_shape(5, 3) == (5, 3)
