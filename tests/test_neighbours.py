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
        """A rebuild must carry the tangential spring of a pair that is still in the list."""
        positions = np.array([[0.0, 0.0], [0.019, 0.0], [0.5, 0.5]])
        radii = np.full(3, 0.01)
        neighbours = NeighbourList(skin=0.002)
        neighbours.update(positions, radii)
        assert neighbours.first.tolist() == [0] and neighbours.second.tolist() == [1]
        neighbours.spring[0] = [1.0e-6, 2.0e-6]
        moved = positions + np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.01]])
        neighbours.update(moved, radii)
        assert neighbours.builds == 2
        assert_allclose(neighbours.spring, [[1.0e-6, 2.0e-6]])

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
        neighbours.remove_particles(np.array([False, True, True]))
        assert neighbours.first.tolist() == [0] and neighbours.second.tolist() == [1]
        assert_allclose(neighbours.spring, [[2.0, 0.0]])

    def test_negative_skin_is_rejected(self) -> None:
        """A negative skin must be rejected."""
        with pytest.raises(ValueError, match=re.escape("skin must be non-negative, got -0.1.")):
            NeighbourList(skin=-0.1)
