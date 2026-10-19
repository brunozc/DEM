"""Tests of the polygonal domain, the boundary conditions and the platen state."""

import re

import numpy as np
import pytest
from numpy.testing import assert_allclose

from DEMcookies.boundary import LoadedPlaten, Open, Platen, Wall
from DEMcookies.geometry import Geometry, project_onto_segments
from DEMcookies.loads import Heaviside


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

    def test_non_positive_platen_mass_is_rejected(self) -> None:
        """A platen needs a positive mass."""
        with pytest.raises(ValueError, match=re.escape("LoadedPlaten mass must be positive, got 0.0.")):
            LoadedPlaten(load=Heaviside(1.0), mass=0.0)

    def test_negative_platen_friction_is_rejected(self) -> None:
        """A platen friction coefficient cannot be negative."""
        with pytest.raises(ValueError, match=re.escape("friction must be non-negative, got -1.0.")):
            LoadedPlaten(load=Heaviside(1.0), mass=1.0, friction=-1.0)


class TestPlaten:
    """Tests of the runtime platen state."""

    @staticmethod
    def _top_platen() -> Platen:
        """Return a platen on the top edge of a unit box, facing down."""
        spec = LoadedPlaten(load=Heaviside(2.0), mass=0.5)
        return Platen(2, spec, np.array([1.0, 1.0]), np.array([0.0, 1.0]), np.array([0.0, -1.0]))

    def test_seat_touches_highest_particle_below(self) -> None:
        """Seating must lower the plate until it touches the highest particle under it, not one beside it."""
        platen = self._top_platen()
        positions = np.array([[0.5, 0.3], [0.2, 0.5], [1.5, 0.95]])
        radii = np.array([0.1, 0.05, 0.1])
        platen.seat(positions, radii)
        assert_allclose(platen.start[1], 0.55)
        assert_allclose(platen.offset, 0.45)

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
        heavy = Platen(2, LoadedPlaten(load=Heaviside(2.0), mass=0.5, self_weight=True), np.array([1.0, 1.0]),
                       np.array([0.0, 1.0]), np.array([0.0, -1.0]))
        assert_allclose(heavy.acceleration(1.0, np.array([0.0, -9.81])), (2.0 + 0.5 * 9.81) / 0.5)
