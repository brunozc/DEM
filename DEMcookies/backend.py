"""Choice of the device that measures the overlaps of polytopes: the CPU, with NumPy, or a GPU, with CuPy.

Measuring how polygons and polyhedra overlap is most of the work of a simulation with shaped particles, and it is made
of large array operations on whole batches of contacts, which a GPU runs many times faster than a CPU. A simulation
on the device "gpu" keeps the shapes of its particles on the GPU and measures the overlaps there with CuPy, the GPU
counterpart of NumPy, while everything else stays on the CPU. CuPy is an optional dependency, which the extra ``gpu``
of the package installs; the device "cpu", the default, needs NumPy only.

The functions of ``DEMcookies.overlap`` and ``DEMcookies.polytopes`` work on NumPy and CuPy arrays alike: each uses
the array module of its arguments, which ``array_module`` returns.
"""

import contextlib
import importlib
from types import ModuleType
from typing import Any, ContextManager

import numpy as np
import numpy.typing as npt

# devices a simulation can measure the overlaps of its polytopes on
DEVICES = ("cpu", "gpu")


def _on_gpu(array: Any) -> bool:
    """Whether an array is a CuPy array, decided without importing CuPy."""
    return type(array).__module__.split(".")[0] == "cupy"


def array_module(*arrays: Any) -> ModuleType:
    """Return the array module of some arrays: CuPy if any of them lives on a GPU, NumPy otherwise.

    Args:
        arrays: NumPy or CuPy arrays.

    Returns:
        The module, ``numpy`` or ``cupy``.
    """
    if any(_on_gpu(array) for array in arrays):
        return gpu_module()
    return np


def gpu_module() -> ModuleType:
    """Import CuPy and check that it can reach a GPU.

    Returns:
        The ``cupy`` module.

    Raises:
        RuntimeError: if CuPy is not installed, or finds no GPU.
    """
    try:
        cupy = importlib.import_module("cupy")
    except ImportError as error:
        raise RuntimeError("The device 'gpu' needs CuPy; install it with pip install 'DEM-cookies[gpu]', or a CuPy "
                           "build for your CUDA version.") from error
    try:
        count = cupy.cuda.runtime.getDeviceCount()
    except Exception as error:
        raise RuntimeError(f"CuPy finds no usable GPU: {error}") from error
    if count < 1:
        raise RuntimeError("CuPy finds no GPU.")
    return cupy


def check_device(device: str) -> None:
    """Check the name of a device.

    Args:
        device: name of the device.

    Raises:
        ValueError: if the device is not one of DEVICES.
    """
    if device not in DEVICES:
        raise ValueError(f"device must be one of {DEVICES}, got {device!r}.")


def to_device(array: npt.ArrayLike, xp: ModuleType) -> Any:
    """Return an array in the memory of the device of an array module.

    Args:
        array: NumPy array, or any array-like.
        xp: array module of the device, ``numpy`` or ``cupy``.

    Returns:
        The array, copied to the GPU for ``cupy``; a NumPy array is returned as it is for ``numpy``.
    """
    return xp.asarray(array)


def to_host(array: Any) -> Any:
    """Return an array in the memory of the CPU, as a NumPy array.

    Args:
        array: NumPy or CuPy array.

    Returns:
        The array, copied from the GPU if it lives there, and as it is otherwise.
    """
    return array.get() if _on_gpu(array) else array


def quiet(xp: ModuleType) -> ContextManager[Any]:
    """Return a context in which divisions by zero go unreported, as they are meant to in the clipping of lines.

    Args:
        xp: array module, ``numpy`` or ``cupy``, which never reports them.

    Returns:
        The context.
    """
    if xp is np:
        return np.errstate(divide="ignore", invalid="ignore")
    return contextlib.nullcontext()
