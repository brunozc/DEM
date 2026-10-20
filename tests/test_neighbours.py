"""Tests of the cell grid and the Verlet neighbour list."""

import re
from typing import Set, Tuple

import numpy as np
import numpy.typing as npt
import pytest
from numpy.testing import assert_allclose

from DEMcookies.neighbours import NeighbourList, find_close_pairs


def _brute_force_pairs(positions: npt.NDArray[np.float64], radii: npt.NDArray[np.float64],
                       skin: float) -> Set[Tuple[int, int]]:
    """Return every pair (i, j), i < j, whose surface gap is below the skin, by testing all pairs."""
    pairs = set()
    for i in range(len(radii)):
        for j in range(i + 1, len(radii)):
            if np.hypot(*(positions[i] - positions[j])) < radii[i] + radii[j] + skin:
                pairs.add((i, j))
    return pairs


class TestFindClosePairs:
    """Tests of find_close_pairs."""

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_matches_brute_force(self, seed: int) -> None:
        """The cell grid must find exactly the pairs an all-pairs search finds."""
        rng = np.random.default_rng(seed)
        positions = rng.uniform(-0.1, 0.3, (300, 2))
        radii = rng.uniform(0.002, 0.012, 300)
        first, second = find_close_pairs(positions, radii, 0.003)
        found = {(int(i), int(j)) for i, j in zip(first, second)}
        assert found == _brute_force_pairs(positions, radii, 0.003)
        assert np.all(first < second)
        keys = first * 1000 + second
        assert np.all(np.diff(keys) > 0)

    def test_fewer_than_two_particles(self) -> None:
        """With fewer than two particles there are no pairs."""
        first, second = find_close_pairs(np.zeros((1, 2)), np.ones(1), 0.1)
        assert first.size == 0 and second.size == 0

    def test_non_finite_positions_are_reported(self) -> None:
        """A blown-up simulation must be reported instead of silently producing garbage cells."""
        positions = np.array([[0.0, 0.0], [np.nan, 1.0]])
        with pytest.raises(FloatingPointError, match=re.escape("Particle positions are no longer finite")):
            find_close_pairs(positions, np.ones(2), 0.1)


class TestNeighbourList:
    """Tests of NeighbourList."""

    def test_springs_survive_rebuild(self) -> None:
        """A rebuild must carry the tangential and rolling springs of a pair that is still in the list."""
        positions = np.array([[0.0, 0.0], [0.019, 0.0], [0.5, 0.5]])
        radii = np.full(3, 0.01)
        neighbours = NeighbourList(skin=0.002)
        neighbours.update(positions, radii)
        assert neighbours.first.tolist() == [0] and neighbours.second.tolist() == [1]
        neighbours.spring[0] = [1.0e-6, 2.0e-6]
        neighbours.rolling[0] = 3.0e-4
        moved = positions + np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.01]])
        neighbours.update(moved, radii)
        assert neighbours.builds == 2
        assert_allclose(neighbours.spring, [[1.0e-6, 2.0e-6]])
        assert_allclose(neighbours.rolling, [3.0e-4])

    def test_small_motion_reuses_list(self) -> None:
        """Motion of less than half the skin must not trigger a rebuild."""
        positions = np.array([[0.0, 0.0], [0.019, 0.0]])
        neighbours = NeighbourList(skin=0.002)
        neighbours.update(positions, np.full(2, 0.01))
        neighbours.update(positions + 0.0004, np.full(2, 0.01))
        assert neighbours.builds == 1

    def test_remove_particles_renumbers_pairs(self) -> None:
        """Removing a particle must drop its pairs and renumber the others, keeping their springs."""
        positions = np.array([[0.0, 0.0], [0.019, 0.0], [0.038, 0.0]])
        neighbours = NeighbourList(skin=0.002)
        neighbours.update(positions, np.full(3, 0.01))
        neighbours.spring[:] = [[1.0, 0.0], [2.0, 0.0]]
        neighbours.rolling[:] = [0.1, 0.2]
        neighbours.remove_particles(np.array([False, True, True]))
        assert neighbours.first.tolist() == [0] and neighbours.second.tolist() == [1]
        assert_allclose(neighbours.spring, [[2.0, 0.0]])
        assert_allclose(neighbours.rolling, [0.2])

    def test_negative_skin_is_rejected(self) -> None:
        """A negative skin must be rejected."""
        with pytest.raises(ValueError, match=re.escape("skin must be non-negative, got -0.1.")):
            NeighbourList(skin=-0.1)


class TestFindClosePairs3D:
    """Tests of find_close_pairs in 3D."""

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_matches_brute_force(self, seed: int) -> None:
        """The cubic cell grid must find exactly the pairs an all-pairs search finds."""
        rng = np.random.default_rng(seed)
        positions = rng.uniform(-0.1, 0.2, (400, 3))
        radii = rng.uniform(0.004, 0.015, 400)
        first, second = find_close_pairs(positions, radii, 0.003)
        found = {(int(i), int(j)) for i, j in zip(first, second)}
        gap = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2) - radii[:, None] - radii[None, :]
        expected = {(int(i), int(j)) for i, j in zip(*np.nonzero(np.triu(gap < 0.003, k=1)))}
        assert found == expected
        assert np.all(first < second)
        assert np.all(np.diff(first * 1000 + second) > 0)

    def test_lattice_neighbours_across_cells(self) -> None:
        """On a cubic lattice spanning many cells every particle must be paired with its six neighbours only."""
        radius, count = 0.01, 6
        axis = np.arange(count) * 2.0 * radius * 0.999
        positions = np.array([(x, y, z) for x in axis for y in axis for z in axis])
        first, second = find_close_pairs(positions, np.full(positions.shape[0], radius), 0.0)
        assert first.size == 3 * count * count * (count - 1)
        assert_allclose(np.linalg.norm(positions[first] - positions[second], axis=1), 2.0 * radius * 0.999)


class TestNeighbourList3D:
    """Tests of NeighbourList in 3D."""

    def test_vector_springs_survive_rebuild(self) -> None:
        """A rebuild must carry the tangential and rolling spring vectors of a pair that is still in the list."""
        positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.019], [0.5, 0.5, 0.5]])
        radii = np.full(3, 0.01)
        neighbours = NeighbourList(skin=0.002, dimension=3)
        neighbours.update(positions, radii)
        assert neighbours.spring.shape == (1, 3) and neighbours.rolling.shape == (1, 3)
        neighbours.spring[0] = [1.0e-6, 2.0e-6, 3.0e-6]
        neighbours.rolling[0] = [3.0e-4, 0.0, -1.0e-4]
        neighbours.update(positions + np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.01, 0.0]]), radii)
        assert neighbours.builds == 2
        assert_allclose(neighbours.spring, [[1.0e-6, 2.0e-6, 3.0e-6]])
        assert_allclose(neighbours.rolling, [[3.0e-4, 0.0, -1.0e-4]])

    def test_invalid_dimension_is_rejected(self) -> None:
        """Only planes and space are supported."""
        with pytest.raises(ValueError, match=re.escape("dimension must be 2 or 3, got 4.")):
            NeighbourList(skin=0.1, dimension=4)
