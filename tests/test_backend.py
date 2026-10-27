"""Tests of the choice of the device that measures the overlaps of polytopes, on the CPU and, if there is one, a GPU."""

import importlib.util
import pickle
import re

import numpy as np
import pytest
from numpy.testing import assert_allclose

from DEMcookies import ConvexPolyhedron, DEMSimulation, Geometry3D, MaterialProperties, Particle3D
from DEMcookies.backend import array_module, check_device, gpu_module, to_device, to_host
from DEMcookies.overlap import Polyhedra, polyhedron_overlap, polyhedron_region_overlap
from DEMcookies.polytopes import PolytopeTable, place, random_orientations


def _has_gpu() -> bool:
    """Whether CuPy is installed and finds a GPU."""
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        gpu_module()
    except RuntimeError:
        return False
    return True


NEEDS_GPU = pytest.mark.skipif(not _has_gpu(), reason="CuPy and a GPU are needed")
MATERIAL = MaterialProperties(density=2500.0, youngs_modulus=5.0e6, poisson_ratio=0.25, restitution=0.3, friction=0.5)


def _stones(count: int, seed: int = 0) -> Polyhedra:
    """Return random stones placed near the origin in random orientations, as a batch of polyhedra on the CPU."""
    rng = np.random.default_rng(seed)
    shapes = [ConvexPolyhedron.random(12, seed=k, aspect=(1.0 + 0.1 * k, 1.0, 0.8)) for k in range(4)]
    table = PolytopeTable.build(3, [shapes[k] for k in rng.integers(4, size=count)], rng.uniform(0.8, 1.2, count))
    positions = rng.uniform(-0.6, 0.6, (count, 3))
    pose = place(table, np.arange(count), positions, random_orientations(rng, count, 3))
    return Polyhedra(pose.vertices, pose.normals, pose.offsets, table.edges, table.edge_faces, table.faces)


def _sim(device: str = "cpu") -> DEMSimulation:
    """Return two polyhedra and a sphere resting in a box, measured on a device."""
    particles = [
        Particle3D(0.03, 0.01, 0.05, 0.01, shape=ConvexPolyhedron.box(1.0, 1.0, 1.0)),
        Particle3D(0.03, 0.03, 0.05, 0.01, shape=ConvexPolyhedron.random(10, seed=3)),
        Particle3D(0.07, 0.01, 0.05, 0.01)
    ]
    return DEMSimulation(Geometry3D.box(0.1, 0.1, 0.1), MATERIAL, particles=particles, device=device)


class TestCPU:
    """Tests of the device "cpu", which needs NumPy only."""

    def test_numpy_arrays_stay_on_the_cpu(self) -> None:
        """A NumPy array must keep NumPy as its module and be left as it is."""
        array = np.arange(3.0)
        assert array_module(array) is np
        assert to_device(array, np) is array
        assert to_host(array) is array

    def test_unknown_device_is_rejected(self) -> None:
        """A device other than cpu or gpu must be rejected, by the check and by a simulation."""
        with pytest.raises(ValueError, match=re.escape("device must be one of ('cpu', 'gpu'), got 'tpu'.")):
            check_device("tpu")
        with pytest.raises(ValueError, match=re.escape("device must be one of ('cpu', 'gpu'), got 'tpu'.")):
            _sim("tpu")

    @pytest.mark.skipif(importlib.util.find_spec("cupy") is not None, reason="CuPy is installed")
    def test_gpu_without_cupy_is_rejected(self) -> None:
        """Asking for the GPU without CuPy must say how to install it."""
        with pytest.raises(RuntimeError, match=re.escape("The device 'gpu' needs CuPy")):
            _sim("gpu")

    def test_device_can_be_changed_and_is_pickled(self) -> None:
        """The device of a simulation must be kept by pickling, and a simulation must step on after a change."""
        sim = _sim()
        sim.rest(0.002)
        copy = pickle.loads(pickle.dumps(sim))
        assert copy.device == "cpu"
        copy.device = "cpu"
        copy.rest(0.002)
        sim.rest(0.002)
        assert_allclose(copy.pos, sim.pos, rtol=0.0, atol=0.0)


@NEEDS_GPU
class TestGPU:
    """Tests of the device "gpu", which need CuPy and a GPU."""

    def test_overlaps_on_the_gpu_match_the_cpu(self) -> None:
        """The overlaps of a batch of stones measured on the GPU must match those measured on the CPU."""
        cupy = gpu_module()
        first, second = _stones(300, seed=1), _stones(300, seed=2)
        on_cpu = polyhedron_overlap(first, second)
        on_gpu = polyhedron_overlap(*(Polyhedra(*(cupy.asarray(getattr(b, name))
                                                  for name in Polyhedra.__dataclass_fields__))
                                      for b in (first, second)))
        assert int(np.count_nonzero(on_cpu.measure)) > 50
        assert_allclose(to_host(on_gpu.measure), on_cpu.measure, rtol=1e-10, atol=1e-14)
        assert_allclose(to_host(on_gpu.surface), on_cpu.surface, rtol=1e-10, atol=1e-14)

    def test_region_overlaps_on_the_gpu_match_the_cpu(self) -> None:
        """The overlaps of stones with the region behind a floor must match on the GPU and on the CPU."""
        cupy = gpu_module()
        stones = _stones(200, seed=3)
        normals = np.tile([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]], (200, 1, 1))
        offsets = np.tile([0.0, 2.0, 2.0, 2.0], (200, 1))
        on_cpu = polyhedron_region_overlap(stones, normals, offsets)
        on_gpu = polyhedron_region_overlap(
            Polyhedra(*(cupy.asarray(getattr(stones, name)) for name in Polyhedra.__dataclass_fields__)),
            cupy.asarray(normals), cupy.asarray(offsets))
        assert int(np.count_nonzero(on_cpu.measure)) > 20
        assert_allclose(to_host(on_gpu.measure), on_cpu.measure, rtol=1e-10, atol=1e-14)

    def test_simulation_on_the_gpu_follows_the_cpu(self) -> None:
        """A simulation measured on the GPU must follow the same one on the CPU to rounding."""
        on_cpu, on_gpu = _sim("cpu"), _sim("gpu")
        on_cpu.rest(0.05)
        on_gpu.rest(0.05)
        assert_allclose(on_gpu.pos, on_cpu.pos, atol=1e-9)
        assert_allclose(on_gpu.force, on_cpu.force, atol=1e-9 * float(np.abs(on_cpu.force).max()))

    def test_gpu_checkpoint_carries_on_on_the_cpu(self) -> None:
        """A simulation pickled on the GPU must hold nothing of the GPU and carry on on the CPU."""
        sim = _sim("gpu")
        sim.rest(0.01)
        copy = pickle.loads(pickle.dumps(sim))
        assert copy.device == "gpu"
        copy.device = "cpu"
        copy.rest(0.01)
        sim.rest(0.01)
        assert_allclose(copy.pos, sim.pos, atol=1e-9)
