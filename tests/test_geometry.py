"""Tests of the polygonal domain, the boundary conditions and the platen state."""

import math
import re
from typing import Optional, Tuple

import numpy as np
import pytest
from numpy.testing import assert_allclose

from DEMcookies.boundary import (LoadedPlaten, Open, Platen, PrescribedPlaten, Wall, resolve_body_direction,
                                 resolve_direction)
from DEMcookies.geometry import Geometry, project_onto_segments
from DEMcookies.loads import Heaviside, Sinusoidal, TimeHistory


class TestGeometry:
    """Tests of Geometry."""

    def test_rectangle_defaults(self) -> None:
        """A rectangle must be walled on three sides, open at the top, and have inward normals."""
        box = Geometry.rectangle(0.2, 0.6)
        assert [type(c) for c in box.boundary_conditions] == [Wall, Wall, Open, Wall]
        assert_allclose(box.inward_normals, [[0.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [1.0, 0.0]], atol=1e-15)
        assert_allclose(box.area, 0.12)
        assert_allclose(box.bounding_box, (0.0, 0.0, 0.2, 0.6))

    def test_clockwise_vertices_keep_inward_normals(self) -> None:
        """Normals must point into the domain whatever the orientation of the vertices."""
        clockwise = Geometry([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)], [Wall()] * 4)
        assert_allclose(clockwise.inward_normals, [[1.0, 0.0], [0.0, -1.0], [-1.0, 0.0], [0.0, 1.0]], atol=1e-15)

    def test_contains_non_convex_polygon(self) -> None:
        """Points in the notch of an L-shaped domain must be outside it."""
        shape = Geometry([(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)], [Wall()] * 6)
        inside = shape.contains(np.array([[0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [1.5, 1.5], [3.0, 0.5]]))
        assert inside.tolist() == [True, True, True, False, False]

    def test_distance_to_edges(self) -> None:
        """Distances must be measured to the closest point of each finite edge."""
        box = Geometry.rectangle(1.0, 1.0)
        expected = [[0.5, 0.75, 0.5, 0.25], [np.sqrt(5.0), np.sqrt(2.0), np.sqrt(2.0), np.sqrt(5.0)]]
        assert_allclose(box.distance_to_edges(np.array([[0.25, 0.5], [2.0, 2.0]])), expected)

    def test_project_onto_segments(self) -> None:
        """Projections beyond a segment end must be clamped to that end."""
        t, offset = project_onto_segments(np.array([[-1.0, 1.0], [0.5, 1.0]]), np.array([[0.0, 0.0]]),
                                          np.array([[1.0, 0.0]]))
        assert_allclose(t, [[0.0], [0.5]])
        assert_allclose(offset, [[[-1.0, 1.0]], [[0.0, 1.0]]])

    def test_too_few_vertices_are_rejected(self) -> None:
        """A polygon needs at least three vertices."""
        with pytest.raises(ValueError, match=re.escape("A polygon needs at least three vertices, got 2.")):
            Geometry([(0, 0), (1, 0)], [Wall(), Wall()])

    def test_malformed_vertices_are_rejected(self) -> None:
        """Vertices must be (x, y) pairs."""
        with pytest.raises(ValueError, match=re.escape("vertices must be a sequence of (x, y) pairs.")):
            Geometry([(0, 0, 0), (1, 0, 0), (1, 1, 0)], [Wall()] * 3)

    def test_boundary_condition_count_is_checked(self) -> None:
        """Every edge needs exactly one boundary condition."""
        with pytest.raises(ValueError, match=re.escape("Expected one boundary condition per edge (4), got 3.")):
            Geometry([(0, 0), (1, 0), (1, 1), (0, 1)], [Wall()] * 3)

    def test_zero_length_edge_is_rejected(self) -> None:
        """A repeated vertex creates a zero-length edge and must be rejected."""
        with pytest.raises(ValueError, match=re.escape("Edge 1 has zero length.")):
            Geometry([(0, 0), (1, 0), (1, 0), (0, 1)], [Wall()] * 4)

    def test_zero_area_is_rejected(self) -> None:
        """Collinear vertices enclose no area and must be rejected."""
        with pytest.raises(ValueError, match=re.escape("The polygon has zero area.")):
            Geometry([(0, 0), (1, 0), (2, 0)], [Wall()] * 3)

    def test_self_intersection_is_rejected(self) -> None:
        """A bow-tie polygon must be rejected."""
        with pytest.raises(ValueError, match=re.escape("Edges 0 and 2 intersect; the polygon must be simple.")):
            Geometry([(0, 0), (2, 2), (2, 0), (0, 1)], [Wall()] * 4)

    def test_fold_back_is_rejected(self) -> None:
        """An edge that doubles back along the previous one must be rejected."""
        with pytest.raises(ValueError, match=re.escape("Edges 1 and 2 fold back onto each other.")):
            Geometry([(0, 0), (2, 0), (2, 2), (2, 1), (0, 1)], [Wall()] * 5)

    def test_edges_sharing_a_platen_form_one_body(self) -> None:
        """Edges that carry the same LoadedPlaten object must be grouped into one body, in edge order."""
        wedge = LoadedPlaten(load=Heaviside(1.0), mass=1.0)
        vertices = [(0, 0), (1, 0), (1, 1), (0.6, 1), (0.5, 0.8), (0.4, 1), (0, 1)]
        shape = Geometry(vertices, [Wall(), Wall(), Open(), wedge, wedge, Open(), Wall()])
        assert len(shape.platen_groups) == 1
        assert shape.platen_groups[0][0] is wedge
        assert shape.platen_groups[0][1] == (3, 4)

    def test_equal_but_separate_platens_are_separate_bodies(self) -> None:
        """Two LoadedPlaten objects with equal fields must still be two bodies."""
        left = LoadedPlaten(load=Heaviside(1.0), mass=1.0)
        right = LoadedPlaten(load=Heaviside(1.0), mass=1.0)
        vertices = [(0, 0), (1, 0), (1, 1), (0.6, 1), (0.4, 1), (0, 1)]
        shape = Geometry(vertices, [Wall(), Wall(), right, Open(), left, Wall()])
        assert [edges for _, edges in shape.platen_groups] == [(2, ), (4, )]

    def test_loaded_and_prescribed_platens_are_separate_bodies(self) -> None:
        """Edges sharing a PrescribedPlaten must form one body, apart from a LoadedPlaten in the same domain."""
        loaded = LoadedPlaten(load=Heaviside(1.0), mass=1.0)
        wedge = PrescribedPlaten(displacement=Heaviside(1.0e-3, rise_time=1.0))
        vertices = [(0, 0), (1, 0), (1, 1), (0.8, 1), (0.7, 0.8), (0.6, 1), (0.4, 1), (0.2, 1), (0, 1)]
        shape = Geometry(vertices, [Wall(), Wall(), Open(), wedge, wedge, Open(), loaded, Open(), Wall()])
        assert shape.platen_groups == ((wedge, (3, 4)), (loaded, (6, )))

    def test_cancelling_normals_are_rejected(self) -> None:
        """A body on opposite sides of the domain has no default direction and must be rejected."""
        both = LoadedPlaten(load=Heaviside(1.0), mass=1.0)
        message = "The inward normals of the platen edges cancel out; give the LoadedPlaten an explicit direction."
        with pytest.raises(ValueError, match=re.escape(message)):
            Geometry.rectangle(1.0, 1.0, bottom=both, top=both)

    def test_cancelling_normals_of_prescribed_body_are_rejected(self) -> None:
        """A prescribed body on opposite sides of the domain has no default direction either."""
        both = PrescribedPlaten(displacement=Heaviside(1.0e-3, rise_time=1.0))
        message = "The inward normals of the platen edges cancel out; give the PrescribedPlaten an explicit direction."
        with pytest.raises(ValueError, match=re.escape(message)):
            Geometry.rectangle(1.0, 1.0, bottom=both, top=both)

    def test_non_positive_rectangle_is_rejected(self) -> None:
        """A rectangle needs a positive width and height."""
        with pytest.raises(ValueError,
                           match=re.escape("Rectangle width and height must be positive, got 0.0 and 1.0.")):
            Geometry.rectangle(0.0, 1.0)


class TestBoundaryConditions:
    """Tests of the boundary condition specifications."""

    def test_negative_wall_friction_is_rejected(self) -> None:
        """A wall friction coefficient cannot be negative."""
        with pytest.raises(ValueError, match=re.escape("friction must be non-negative, got -0.2.")):
            Wall(friction=-0.2)

    def test_zero_platen_direction_is_rejected(self) -> None:
        """A platen cannot move along a zero vector."""
        with pytest.raises(ValueError, match=re.escape("direction must be a finite, non-zero (x, y) vector")):
            LoadedPlaten(load=Heaviside(1.0), mass=1.0, direction=(0.0, 0.0))

    def test_default_direction_is_length_weighted_normal(self) -> None:
        """Without a direction, a body moves along the length-weighted mean of the inward normals of its edges."""
        spec = LoadedPlaten(load=Heaviside(1.0), mass=1.0)
        starts = np.array([[0.0, 0.0], [2.0, 0.0]])
        ends = np.array([[2.0, 0.0], [2.0, 1.0]])
        normals = np.array([[0.0, 1.0], [-1.0, 0.0]])
        assert_allclose(resolve_direction(spec, starts, ends, normals), np.array([-1.0, 2.0]) / np.sqrt(5.0))

    def test_explicit_direction_is_normalised(self) -> None:
        """An explicit direction is used as given, scaled to unit length."""
        spec = LoadedPlaten(load=Heaviside(1.0), mass=1.0, direction=(3.0, -4.0))
        direction = resolve_direction(spec, np.zeros((1, 2)), np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]]))
        assert_allclose(direction, [0.6, -0.8])

    def test_non_positive_platen_mass_is_rejected(self) -> None:
        """A platen needs a positive mass."""
        with pytest.raises(ValueError, match=re.escape("LoadedPlaten mass must be positive, got 0.0.")):
            LoadedPlaten(load=Heaviside(1.0), mass=0.0)

    def test_negative_platen_friction_is_rejected(self) -> None:
        """A platen friction coefficient cannot be negative."""
        with pytest.raises(ValueError, match=re.escape("friction must be non-negative, got -1.0.")):
            LoadedPlaten(load=Heaviside(1.0), mass=1.0, friction=-1.0)

    def test_smooth_prescribed_start_is_accepted(self) -> None:
        """A sinusoid shifted to start from zero is a valid displacement history."""
        spec = PrescribedPlaten(displacement=Sinusoidal(1.0e-3, 2.0, mean=1.0e-3, phase=-0.5 * math.pi))
        assert_allclose(spec.displacement(0.25), 2.0e-3)

    def test_prescribed_displacement_must_start_at_zero(self) -> None:
        """A displacement history that does not start from zero would make the body jump, and must be rejected."""
        message = "PrescribedPlaten displacement must be zero when loading starts, got 0.001 m."
        with pytest.raises(ValueError, match=re.escape(message)):
            PrescribedPlaten(displacement=Heaviside(1.0e-3))

    def test_undefined_prescribed_start_is_rejected(self) -> None:
        """A displacement history that is not a number when loading starts must be rejected."""
        message = "PrescribedPlaten displacement must be zero when loading starts, got nan m."
        with pytest.raises(ValueError, match=re.escape(message)):
            PrescribedPlaten(displacement=lambda time: math.nan)

    def test_negative_prescribed_friction_is_rejected(self) -> None:
        """A prescribed platen friction coefficient cannot be negative."""
        with pytest.raises(ValueError, match=re.escape("friction must be non-negative, got -1.0.")):
            PrescribedPlaten(displacement=Heaviside(1.0e-3, rise_time=1.0), friction=-1.0)

    def test_zero_prescribed_direction_is_rejected(self) -> None:
        """A prescribed platen cannot move along a zero vector."""
        message = "PrescribedPlaten direction must be a finite, non-zero (x, y) vector, got (0.0, 0.0)."
        with pytest.raises(ValueError, match=re.escape(message)):
            PrescribedPlaten(displacement=Heaviside(1.0e-3, rise_time=1.0), direction=(0.0, 0.0))


class TestPlaten:
    """Tests of the runtime platen state."""

    @staticmethod
    def _top_platen(direction: Optional[Tuple[float, float]] = None, self_weight: bool = False) -> Platen:
        """Return a platen on the top edge of a unit box, facing down."""
        spec = LoadedPlaten(load=Heaviside(2.0), mass=0.5, direction=direction, self_weight=self_weight)
        return Platen((2, ), spec, np.array([[1.0, 1.0]]), np.array([[0.0, 1.0]]), np.array([[0.0, -1.0]]))

    def test_seat_touches_highest_particle_below(self) -> None:
        """Seating must lower the plate until it touches the highest particle under it, not one beside it."""
        platen = self._top_platen()
        positions = np.array([[0.5, 0.3], [0.2, 0.5], [1.5, 0.95]])
        radii = np.array([0.1, 0.05, 0.1])
        platen.seat(positions, radii)
        assert_allclose(platen.starts[0, 1], 0.55)
        assert_allclose(platen.offset, 0.45)

    def test_seat_ignores_particles_behind(self) -> None:
        """A particle behind the plate, outside the domain, must neither block nor attract it."""
        platen = self._top_platen()
        platen.seat(np.array([[0.5, 1.05], [0.5, 0.3]]), np.array([0.1, 0.1]))
        assert_allclose(platen.offset, 0.6)

    def test_seat_along_oblique_direction(self) -> None:
        """A plate moving obliquely must stop where its swept edge first touches the particle."""
        platen = self._top_platen(direction=(0.6, -0.8))
        platen.seat(np.array([[0.5, 0.3]]), np.array([0.1]))
        # the line closes in at 0.8 per unit of travel, so the 0.6 gap closes after 0.75
        assert_allclose(platen.offset, 0.75)
        assert_allclose(platen.starts[0], [1.0 + 0.6 * 0.75, 1.0 - 0.8 * 0.75])

    def test_seat_never_moves_backwards(self) -> None:
        """A particle already overlapping the plate must not push it out of the domain."""
        platen = self._top_platen()
        platen.seat(np.array([[0.5, 0.95]]), np.array([0.1]))
        assert_allclose(platen.offset, 0.0)

    def test_displacement_is_measured_from_activation(self) -> None:
        """The reported displacement must start from zero when loading starts."""
        platen = self._top_platen()
        platen.seat(np.array([[0.5, 0.3]]), np.array([0.1]))
        platen.activate()
        platen.offset += 0.01
        assert_allclose(platen.displacement, 0.01)

    def test_acceleration(self) -> None:
        """The plate must accelerate with (load - reaction + optional weight) / mass along its normal."""
        platen = self._top_platen()
        platen.reaction = 0.5
        assert_allclose(platen.acceleration(1.0, np.array([0.0, -9.81])), (2.0 - 0.5) / 0.5)
        heavy = self._top_platen(self_weight=True)
        assert_allclose(heavy.acceleration(1.0, np.array([0.0, -9.81])), (2.0 + 0.5 * 9.81) / 0.5)

    @staticmethod
    def _prescribed_top_platen() -> Platen:
        """Return a platen on the top edge of a unit box that comes down 0.1 per unit of time."""
        spec = PrescribedPlaten(displacement=TimeHistory([0.0, 1.0], [0.0, 0.1]))
        return Platen((2, ), spec, np.array([[1.0, 1.0]]), np.array([[0.0, 1.0]]), np.array([[0.0, -1.0]]))

    def test_drive_follows_the_prescribed_displacement(self) -> None:
        """A prescribed plate must move to its displacement at the end of the step, from where it was seated."""
        platen = self._prescribed_top_platen()
        assert platen.prescribed and not self._top_platen().prescribed
        platen.seat(np.array([[0.5, 0.3]]), np.array([0.1]))
        platen.activate()
        platen.drive(0.5, 0.1)
        assert_allclose(platen.displacement, 0.06)
        assert_allclose(platen.offset, 0.6 + 0.06)
        # the mean velocity over the step
        assert_allclose(platen.velocity, (0.06 - 0.05) / 0.1)
        assert_allclose(platen.starts[0, 1], 1.0 - 0.66)

    def test_applied_load(self) -> None:
        """A loaded plate applies its load, and a prescribed one, which has no mass, its reaction."""
        loaded = self._top_platen()
        loaded.reaction = 0.7
        assert_allclose(loaded.applied_load(1.0), 2.0)
        prescribed = self._prescribed_top_platen()
        prescribed.reaction = 0.7
        assert_allclose(prescribed.applied_load(1.0), 0.7)

    def test_prescribed_plate_has_no_acceleration(self) -> None:
        """A prescribed plate follows its path and has no equation of motion to accelerate it."""
        message = "A PrescribedPlaten follows its displacement history and has no equation of motion."
        with pytest.raises(TypeError, match=re.escape(message)):
            self._prescribed_top_platen().acceleration(1.0, np.array([0.0, -9.81]))

    def test_loaded_plate_cannot_be_driven(self) -> None:
        """A loaded plate moves under its load and has no displacement history to follow."""
        message = "A LoadedPlaten moves under its load and cannot be driven along a displacement."
        with pytest.raises(TypeError, match=re.escape(message)):
            self._top_platen().drive(0.5, 0.1)


class TestLoadedPlatenDirection3D:
    """Tests of the direction of a loaded body in three dimensions."""

    def test_3d_direction_is_normalised(self) -> None:
        """An explicit 3D direction is used as given, scaled to unit length."""
        spec = LoadedPlaten(load=Heaviside(1.0), mass=1.0, direction=(0.0, -3.0, 4.0))
        assert_allclose(resolve_body_direction(spec, np.ones(1), np.array([[0.0, -1.0, 0.0]])), [0.0, -0.6, 0.8])

    @pytest.mark.parametrize("direction, kind", [
        ((0.0, 0.0, 0.0), "(x, y, z)"),
        ((0.0, math.inf, 1.0), "(x, y, z)"),
        ((1.0, 0.0, 0.0, 0.0), "(x, y) or (x, y, z)"),
    ])
    def test_invalid_direction_is_rejected(self, direction: Tuple[float, ...], kind: str) -> None:
        """A direction must be a finite, non-zero vector of two or three components."""
        message = f"LoadedPlaten direction must be a finite, non-zero {kind} vector, got {direction}."
        with pytest.raises(ValueError, match=re.escape(message)):
            LoadedPlaten(load=Heaviside(1.0), mass=1.0, direction=direction)

    def test_3d_direction_in_polygon_is_rejected(self) -> None:
        """A loaded body of a polygon cannot move along a 3D direction."""
        platen = LoadedPlaten(load=Heaviside(1.0), mass=1.0, direction=(0.0, -1.0, 0.0))
        message = "LoadedPlaten direction (0.0, -1.0, 0.0) has 3 components, but the domain is 2-dimensional."
        with pytest.raises(ValueError, match=re.escape(message)):
            Geometry.rectangle(1.0, 1.0, top=platen)
