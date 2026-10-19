"""Tests of the load time functions."""

import math
import re

import pytest
from numpy.testing import assert_allclose

from DEMcookies.loads import Heaviside, Pulse, Sinusoidal, TimeHistory


class TestPulse:
    """Tests of the rectangular pulse."""

    def test_values(self) -> None:
        """The pulse must be on in [start, start + duration) and off elsewhere."""
        pulse = Pulse(amplitude=5.0, duration=0.2, start_time=0.1)
        assert_allclose([pulse(t) for t in (0.0, 0.1, 0.2, 0.299, 0.301, 1.0)], [0.0, 5.0, 5.0, 5.0, 0.0, 0.0])

    def test_non_positive_duration_is_rejected(self) -> None:
        """A pulse without duration must be rejected."""
        with pytest.raises(ValueError, match=re.escape("Pulse duration must be positive, got 0.0.")):
            Pulse(amplitude=1.0, duration=0.0)

    def test_negative_start_is_rejected(self) -> None:
        """Loading time starts at zero, so a negative start must be rejected."""
        with pytest.raises(ValueError, match=re.escape("Pulse start_time must be non-negative, got -1.0.")):
            Pulse(amplitude=1.0, duration=1.0, start_time=-1.0)


class TestHeaviside:
    """Tests of the step load."""

    def test_sudden_step(self) -> None:
        """Without a rise time the load must jump to the amplitude at the start time and stay there."""
        step = Heaviside(amplitude=10.0, start_time=0.5)
        assert_allclose([step(t) for t in (0.0, 0.499, 0.5, 3.0)], [0.0, 0.0, 10.0, 10.0])

    def test_ramped_step(self) -> None:
        """With a rise time the load must grow linearly to the amplitude and then be held."""
        step = Heaviside(amplitude=1000.0, rise_time=0.4)
        assert_allclose([step(t) for t in (0.0, 0.1, 0.4, 1.0)], [0.0, 250.0, 1000.0, 1000.0])

    def test_negative_rise_time_is_rejected(self) -> None:
        """A negative rise time must be rejected."""
        with pytest.raises(ValueError, match=re.escape("Heaviside rise_time must be non-negative, got -0.1.")):
            Heaviside(amplitude=1.0, rise_time=-0.1)

    def test_negative_start_is_rejected(self) -> None:
        """A negative start time must be rejected."""
        with pytest.raises(ValueError, match=re.escape("Heaviside start_time must be non-negative, got -2.0.")):
            Heaviside(amplitude=1.0, start_time=-2.0)


class TestSinusoidal:
    """Tests of the harmonic load."""

    def test_values(self) -> None:
        """The load must be mean + amplitude sin(2 pi f (t - t0) + phase) after the start and zero before."""
        load = Sinusoidal(amplitude=2.0, frequency=5.0, mean=3.0, phase=0.3, start_time=0.1)
        assert_allclose(load(0.05), 0.0)
        assert_allclose(load(0.2), 3.0 + 2.0 * math.sin(2.0 * math.pi * 5.0 * 0.1 + 0.3))

    def test_smooth_start(self) -> None:
        """With mean = amplitude and phase = -pi / 2 the load must start from zero."""
        load = Sinusoidal(amplitude=2.0, frequency=1.0, mean=2.0, phase=-0.5 * math.pi)
        assert_allclose(load(0.0), 0.0, atol=1e-12)
        assert_allclose(load(0.5), 4.0)

    def test_non_positive_frequency_is_rejected(self) -> None:
        """A non-positive frequency must be rejected."""
        with pytest.raises(ValueError, match=re.escape("Sinusoidal frequency must be positive, got 0.0.")):
            Sinusoidal(amplitude=1.0, frequency=0.0)

    def test_negative_start_is_rejected(self) -> None:
        """A negative start time must be rejected."""
        with pytest.raises(ValueError, match=re.escape("Sinusoidal start_time must be non-negative, got -1.0.")):
            Sinusoidal(amplitude=1.0, frequency=1.0, start_time=-1.0)


class TestTimeHistory:
    """Tests of the user-defined load history."""

    def test_interpolation_and_clamping(self) -> None:
        """The table must be interpolated linearly, with the end values held outside it."""
        history = TimeHistory([0.1, 0.3, 0.5], [0.0, 10.0, 4.0])
        assert_allclose([history(t) for t in (0.0, 0.2, 0.3, 0.4, 2.0)], [0.0, 5.0, 10.0, 7.0, 4.0])

    def test_single_point_is_constant(self) -> None:
        """A single point must give a constant load."""
        assert_allclose(TimeHistory([0.0], [7.0])(12.0), 7.0)

    def test_empty_table_is_rejected(self) -> None:
        """A table without points must be rejected."""
        with pytest.raises(ValueError, match=re.escape("TimeHistory needs at least one point.")):
            TimeHistory([], [])

    def test_length_mismatch_is_rejected(self) -> None:
        """Times and values of different lengths must be rejected."""
        with pytest.raises(ValueError, match=re.escape("TimeHistory got 2 times but 3 values.")):
            TimeHistory([0.0, 1.0], [0.0, 1.0, 2.0])

    def test_non_increasing_times_are_rejected(self) -> None:
        """Repeated or decreasing times must be rejected."""
        with pytest.raises(ValueError, match=re.escape("TimeHistory times must be strictly increasing.")):
            TimeHistory([0.0, 1.0, 1.0], [0.0, 1.0, 2.0])

    def test_non_finite_values_are_rejected(self) -> None:
        """Non-finite times or values must be rejected."""
        with pytest.raises(ValueError, match=re.escape("TimeHistory times and values must be finite.")):
            TimeHistory([0.0, 1.0], [0.0, float("nan")])
