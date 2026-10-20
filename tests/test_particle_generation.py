"""Tests of the lattice particle generator."""

import re

import numpy as np
import pytest

from DEMcookies.boundary import Open, Wall
from DEMcookies.geometry import Geometry
from DEMcookies.geometry3d import Geometry3D
from DEMcookies.particle_generation import Particle3D, generate_lattice_particles


class TestGenerateLatticeParticles:
    """Tests of generate_lattice_particles."""

    def test_particles_fit_inside_without_overlap(self) -> None:
        """Particles must lie inside a non-rectangular domain, clear of its edges and of each other."""
        geometry = Geometry([(0.0, 0.0), (0.2, 0.0), (0.3, 0.2), (-0.1, 0.2)], [Wall(), Wall(), Open(), Wall()])
        particles = generate_lattice_particles(geometry, 60, (0.004, 0.008), seed=3)
        positions = np.array([[p.x, p.y] for p in particles])
        radii = np.array([p.radius for p in particles])
        assert len(particles) == 60
        assert np.all(geometry.distance_to_edges(positions).min(axis=1) >= radii)
        assert np.all(geometry.contains(positions))
        delta = positions[:, None, :] - positions[None, :, :]
        gap = np.hypot(delta[..., 0], delta[..., 1]) - (radii[:, None] + radii[None, :])
        np.fill_diagonal(gap, np.inf)
        assert gap.min() >= 0.0
        assert np.all((radii >= 0.004) & (radii <= 0.008))

    def test_lowest_rows_are_filled_first(self) -> None:
        """With few particles only the bottom of the domain must be used."""
        particles = generate_lattice_particles(Geometry.rectangle(0.2, 1.0), 5, (0.01, 0.01), jitter=0.0)
        assert max(p.y for p in particles) < 0.05

    def test_too_many_particles_are_rejected(self) -> None:
        """Requesting more particles than lattice sites must be reported."""
        with pytest.raises(ValueError, match=re.escape("lattice sites fit in the domain, but 1000 particles")):
            generate_lattice_particles(Geometry.rectangle(0.1, 0.1), 1000, (0.005, 0.01))

    def test_invalid_radius_range_is_rejected(self) -> None:
        """The radius range must be positive and ordered."""
        with pytest.raises(ValueError,
                           match=re.escape("radius_range must satisfy 0 < r_min <= r_max, got (0.0, 0.01).")):
            generate_lattice_particles(Geometry.rectangle(0.1, 0.1), 5, (0.0, 0.01))

    def test_invalid_spacing_is_rejected(self) -> None:
        """A spacing factor below one would create overlaps and must be rejected."""
        with pytest.raises(ValueError, match=re.escape("spacing_factor must be at least 1, got 0.9.")):
            generate_lattice_particles(Geometry.rectangle(0.1, 0.1), 5, (0.005, 0.01), spacing_factor=0.9)

    def test_invalid_jitter_is_rejected(self) -> None:
        """The jitter is a fraction of the free space and must lie in [0, 1]."""
        with pytest.raises(ValueError, match=re.escape("jitter must be in [0, 1], got 1.5.")):
            generate_lattice_particles(Geometry.rectangle(0.1, 0.1), 5, (0.005, 0.01), jitter=1.5)


class TestGenerateLatticeParticles3D:
    """Tests of generate_lattice_particles in a polyhedron."""

    def test_particles_fit_inside_without_overlap(self) -> None:
        """Particles must lie inside a non-box polyhedron, clear of its faces and of each other."""
        trapezoid = Geometry([(0.0, 0.0), (0.2, 0.0), (0.3, 0.2), (-0.1, 0.2)], [Wall(), Wall(), Open(), Wall()])
        geometry = Geometry3D.extrude(trapezoid, 0.1)
        particles = generate_lattice_particles(geometry, 80, (0.004, 0.008), seed=3)
        assert len(particles) == 80 and all(isinstance(p, Particle3D) for p in particles)
        positions = np.array([[p.x, p.y, p.z] for p in particles])
        radii = np.array([p.radius for p in particles])
        assert np.all(geometry.contains(positions))
        assert np.all(geometry.distance_to_faces(positions).min(axis=1) >= radii)
        gap = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2) - (radii[:, None] + radii[None, :])
        np.fill_diagonal(gap, np.inf)
        assert gap.min() >= 0.0
        assert np.all((radii >= 0.004) & (radii <= 0.008))

    def test_lowest_layers_are_filled_first(self) -> None:
        """With few particles only the bottom layer of the domain must be used."""
        particles = generate_lattice_particles(Geometry3D.box(0.2, 1.0, 0.2), 20, (0.01, 0.01), jitter=0.0)
        assert max(p.y for p in particles) < 0.05

    def test_too_many_particles_are_rejected(self) -> None:
        """Requesting more particles than lattice sites must be reported."""
        with pytest.raises(ValueError, match=re.escape("lattice sites fit in the domain, but 1000 particles")):
            generate_lattice_particles(Geometry3D.box(0.1, 0.1, 0.1), 1000, (0.005, 0.01))
