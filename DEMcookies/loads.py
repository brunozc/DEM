"""Time functions F(t) for the load applied by a loaded platen, or u(t) for the displacement of a prescribed one.

Every load is a callable that maps the time since loading started [s] to a force [N]. A positive force pushes the
platen into the domain. Besides the classes below, any plain function with that signature can be used. The same
classes describe the displacement history of a ``PrescribedPlaten``, in metres instead of newtons.
"""

import math
from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

import numpy as np

LoadFunction = Callable[[float], float]
DisplacementFunction = Callable[[float], float]


@dataclass(frozen=True)
class Pulse:
    """Rectangular pulse: a constant load during a finite time window and zero outside it.

    Attributes:
        amplitude: load during the pulse [N].
        duration: length of the pulse [s].
        start_time: time at which the pulse starts, measured from the start of loading [s].
    """

    amplitude: float
    duration: float
    start_time: float = 0.0

    def __post_init__(self) -> None:
        """Validate the pulse.

        Raises:
            ValueError: if the duration is not positive.
            ValueError: if the start time is negative.
        """
        if self.duration <= 0.0:
            raise ValueError(f"Pulse duration must be positive, got {self.duration}.")
        if self.start_time < 0.0:
            raise ValueError(f"Pulse start_time must be non-negative, got {self.start_time}.")

    def __call__(self, time: float) -> float:
        """Evaluate the load.

        Args:
            time: time since loading started [s].

        Returns:
            The applied load [N].
        """
        if self.start_time <= time < self.start_time + self.duration:
            return self.amplitude
        return 0.0


@dataclass(frozen=True)
class Heaviside:
    """Step load that switches on at a given time and is then held.

    A non-zero rise time replaces the jump by a linear ramp, so a load can be applied quasi-statically instead of
    suddenly.

    Attributes:
        amplitude: load after the step [N].
        start_time: time at which the step starts, measured from the start of loading [s].
        rise_time: duration of the linear ramp from zero to the amplitude [s]; zero gives a sudden step.
    """

    amplitude: float
    start_time: float = 0.0
    rise_time: float = 0.0

    def __post_init__(self) -> None:
        """Validate the step.

        Raises:
            ValueError: if the start time is negative.
            ValueError: if the rise time is negative.
        """
        if self.start_time < 0.0:
            raise ValueError(f"Heaviside start_time must be non-negative, got {self.start_time}.")
        if self.rise_time < 0.0:
            raise ValueError(f"Heaviside rise_time must be non-negative, got {self.rise_time}.")

    def __call__(self, time: float) -> float:
        """Evaluate the load.

        Args:
            time: time since loading started [s].

        Returns:
            The applied load [N].
        """
        elapsed = time - self.start_time
        if elapsed < 0.0:
            return 0.0
        if elapsed >= self.rise_time:
            return self.amplitude
        return self.amplitude * elapsed / self.rise_time


@dataclass(frozen=True)
class Sinusoidal:
    """Harmonic load history, F(t) = mean + amplitude sin(2 pi f (t - start_time) + phase), zero before it starts.

    A platen cannot pull on cohesionless particles, so cyclic compression is usually given a mean load at least as
    large as the amplitude. Setting ``mean = amplitude`` and ``phase = -pi / 2`` starts the load smoothly from zero.

    Attributes:
        amplitude: amplitude of the oscillation [N].
        frequency: frequency of the oscillation [Hz].
        mean: constant load the oscillation is superimposed on [N].
        phase: phase angle at the start time [rad].
        start_time: time at which the load starts, measured from the start of loading [s].
    """

    amplitude: float
    frequency: float
    mean: float = 0.0
    phase: float = 0.0
    start_time: float = 0.0

    def __post_init__(self) -> None:
        """Validate the harmonic load.

        Raises:
            ValueError: if the frequency is not positive.
            ValueError: if the start time is negative.
        """
        if self.frequency <= 0.0:
            raise ValueError(f"Sinusoidal frequency must be positive, got {self.frequency}.")
        if self.start_time < 0.0:
            raise ValueError(f"Sinusoidal start_time must be non-negative, got {self.start_time}.")

    def __call__(self, time: float) -> float:
        """Evaluate the load.

        Args:
            time: time since loading started [s].

        Returns:
            The applied load [N].
        """
        elapsed = time - self.start_time
        if elapsed < 0.0:
            return 0.0
        return self.mean + self.amplitude * math.sin(2.0 * math.pi * self.frequency * elapsed + self.phase)


@dataclass(frozen=True)
class TimeHistory:
    """User-defined load history, interpolated linearly between the given points.

    Before the first point the first value is used, and after the last point the last value is held.

    Attributes:
        times: strictly increasing times of the points, measured from the start of loading [s].
        values: load at each time [N].
    """

    times: Tuple[float, ...]
    values: Tuple[float, ...]

    def __init__(self, times: Sequence[float], values: Sequence[float]) -> None:
        """Create a load history from a table of points.

        Args:
            times: strictly increasing times of the points, measured from the start of loading [s].
            values: load at each time [N].

        Raises:
            ValueError: if no points are given.
            ValueError: if times and values have different lengths.
            ValueError: if a time or value is not finite.
            ValueError: if the times are not strictly increasing.
        """
        times_array = np.asarray(times, dtype=np.float64).ravel()
        values_array = np.asarray(values, dtype=np.float64).ravel()
        if times_array.size == 0:
            raise ValueError("TimeHistory needs at least one point.")
        if times_array.size != values_array.size:
            raise ValueError(f"TimeHistory got {times_array.size} times but {values_array.size} values.")
        if not (np.all(np.isfinite(times_array)) and np.all(np.isfinite(values_array))):
            raise ValueError("TimeHistory times and values must be finite.")
        if np.any(np.diff(times_array) <= 0.0):
            raise ValueError("TimeHistory times must be strictly increasing.")
        object.__setattr__(self, "times", tuple(float(t) for t in times_array))
        object.__setattr__(self, "values", tuple(float(v) for v in values_array))

    def __call__(self, time: float) -> float:
        """Evaluate the load.

        Args:
            time: time since loading started [s].

        Returns:
            The applied load [N].
        """
        return float(np.interp(time, self.times, self.values))
