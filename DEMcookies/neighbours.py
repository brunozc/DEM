"""Broad-phase contact detection: a uniform cell grid feeding a Verlet neighbour list.

The cell grid finds every pair of particles whose surfaces are closer than a skin distance. The neighbour list keeps
those candidate pairs, together with the tangential and rolling springs of each pair, until some particle has moved
more than half the skin. Until then no pair outside the list can have come into contact, so the list can be reused.
"""

from typing import Tuple

import numpy as np
import numpy.typing as npt

from .globals import HALF_STENCIL_OFFSETS, PAIR_KEY_SHIFT


def pair_keys(first: npt.NDArray[np.int64], second: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """Pack particle pairs into single integers that sort like the pairs themselves.

    Args:
        first: index of the first particle of each pair.
        second: index of the second particle of each pair.

    Returns:
        One key per pair.
    """
    return np.asarray((first.astype(np.int64) << PAIR_KEY_SHIFT) | second.astype(np.int64))


def find_close_pairs(positions: npt.NDArray[np.float64], radii: npt.NDArray[np.float64],
                     skin: float) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Find every pair of particles whose surfaces are less than ``skin`` apart.

    Particles are binned into square cells of side ``2 R_max + skin``, so two close particles always sit in the same
    or in adjacent cells. Each cell is compared with itself and with half of its neighbours, which visits every
    unordered pair of cells exactly once.

    Args:
        positions: particle centres, shape (n, 2) [m].
        radii: particle radii, shape (n,) [m].
        skin: largest surface gap of a returned pair [m].

    Returns:
        The pairs (i, j) with i < j, sorted by i and then by j.

    Raises:
        FloatingPointError: if a particle position is not finite.
    """
    n = positions.shape[0]
    empty = np.zeros(0, dtype=np.int64)
    if n < 2:
        return empty, empty
    if not np.all(np.isfinite(positions)):
        raise FloatingPointError("Particle positions are no longer finite; the time step is too large for the "
                                 "contact stiffness.")

    cell_size = 2.0 * float(radii.max()) + skin
    cell = np.floor((positions - positions.min(axis=0)) / cell_size).astype(np.int64)
    # a padding column on each side keeps the -1 and +1 x offsets from wrapping into the neighbouring row
    n_columns = int(cell[:, 0].max()) + 3
    cell_id = (cell[:, 0] + 1) + cell[:, 1] * n_columns
    order = np.argsort(cell_id, kind="stable")
    sorted_ids = cell_id[order]
    particle = np.arange(n, dtype=np.int64)

    firsts = []
    seconds = []
    for dx, dy in HALF_STENCIL_OFFSETS:
        target = cell_id + dx + dy * n_columns
        begin = np.searchsorted(sorted_ids, target, side="left")
        count = np.searchsorted(sorted_ids, target, side="right") - begin
        total = int(count.sum())
        if total == 0:
            continue
        first = np.repeat(particle, count)
        # position of each candidate within its target cell, i.e. 0..count-1 for every particle
        within = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(count) - count, count)
        second = order[np.repeat(begin, count) + within]
        if dx == 0 and dy == 0:
            keep = first < second
            first, second = first[keep], second[keep]
        firsts.append(first)
        seconds.append(second)
    if not firsts:
        return empty, empty

    first = np.concatenate(firsts)
    second = np.concatenate(seconds)
    i = np.minimum(first, second)
    j = np.maximum(first, second)
    delta = positions[i] - positions[j]
    reach = radii[i] + radii[j] + skin
    close = np.einsum("ij,ij->i", delta, delta) < reach * reach
    i, j = i[close], j[close]
    order = np.argsort(pair_keys(i, j), kind="stable")
    return i[order], j[order]


class NeighbourList:
    """Verlet list of candidate particle pairs, together with the tangential and rolling springs of each pair."""

    def __init__(self, skin: float) -> None:
        """Create an empty list.

        Args:
            skin: distance added to the contact distance of a pair when the list is built [m].

        Raises:
            ValueError: if the skin is negative.
        """
        if skin < 0.0:
            raise ValueError(f"skin must be non-negative, got {skin}.")
        self.skin = skin
        self.first: npt.NDArray[np.int64] = np.zeros(0, dtype=np.int64)
        self.second: npt.NDArray[np.int64] = np.zeros(0, dtype=np.int64)
        self.spring: npt.NDArray[np.float64] = np.zeros((0, 2))
        self.rolling: npt.NDArray[np.float64] = np.zeros(0)
        self.builds = 0
        self.__reference: npt.NDArray[np.float64] = np.zeros((0, 2))
        self.__valid = False

    def invalidate(self) -> None:
        """Force a rebuild at the next update, e.g. after particles were added."""
        self.__valid = False

    def update(self, positions: npt.NDArray[np.float64], radii: npt.NDArray[np.float64]) -> None:
        """Rebuild the list if it may have missed a contact.

        Args:
            positions: particle centres, shape (n, 2) [m].
            radii: particle radii, shape (n,) [m].
        """
        if self.__valid and positions.shape == self.__reference.shape:
            displacement = positions - self.__reference
            max_sq = float(np.max(np.einsum("ij,ij->i", displacement, displacement), initial=0.0))
            # two particles approaching each other close the gap at twice the displacement of either
            if max_sq <= (0.5 * self.skin)**2:
                return
        self.__rebuild(positions, radii)

    def remove_particles(self, keep: npt.NDArray[np.bool_]) -> None:
        """Drop the pairs of removed particles and renumber the others.

        Args:
            keep: True for every particle that stays in the simulation.
        """
        new_index = np.cumsum(keep) - 1
        retained = keep[self.first] & keep[self.second]
        # renumbering preserves the order of the indices, so the pairs stay sorted
        self.first = new_index[self.first[retained]].astype(np.int64)
        self.second = new_index[self.second[retained]].astype(np.int64)
        self.spring = self.spring[retained]
        self.rolling = self.rolling[retained]
        self.__valid = False

    def __rebuild(self, positions: npt.NDArray[np.float64], radii: npt.NDArray[np.float64]) -> None:
        """Rebuild the list and carry the springs of surviving pairs over.

        Args:
            positions: particle centres, shape (n, 2) [m].
            radii: particle radii, shape (n,) [m].
        """
        first, second = find_close_pairs(positions, radii, self.skin)
        spring = np.zeros((first.size, 2))
        rolling = np.zeros(first.size)
        if self.first.size and first.size:
            old_keys = pair_keys(self.first, self.second)
            new_keys = pair_keys(first, second)
            index = np.minimum(np.searchsorted(old_keys, new_keys), old_keys.size - 1)
            found = old_keys[index] == new_keys
            spring[found] = self.spring[index[found]]
            rolling[found] = self.rolling[index[found]]
        self.first = first
        self.second = second
        self.spring = spring
        self.rolling = rolling
        self.builds += 1
        self.__reference = positions.copy()
        self.__valid = True
