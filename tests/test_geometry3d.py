"""Tests of the polyhedral domain and of loaded bodies on its faces."""

import math
import re
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pytest
from numpy.testing import assert_allclose

from DEMcookies.boundary import BoundaryCondition, LoadedPlaten, Open, Platen, Wall
from DEMcookies.geometry import Geometry
from DEMcookies.geometry3d import Geometry3D
from DEMcookies.loads import Heaviside


def _box_faces() -> Tuple[List[List[float]], List[Tuple[int, ...]]]:
    """Return the vertices and faces of the unit cube, as Geometry3D.box numbers them."""
    box = Geometry3D.box(1.0, 1.0, 1.0)
    return box.vertices.tolist(), list(box.faces)


def _walls(count: int) -> List[BoundaryCondition]:
    """Return a wall for each of a number of faces."""
    return [Wall()] * count


class TestGeometry3D:
    """Tests of Geometry3D."""

    def test_box_defaults(self) -> None:
        """A box must be walled on five sides, open at the top, and have inward normals."""
        box = Geometry3D.box(0.2, 0.6, 0.1)
        assert [type(c) for c in box.boundary_conditions] == [Wall, Wall, Open, Wall, Wall, Wall]
        expected = [[0, 1, 0], [-1, 0, 0], [0, -1, 0], [1, 0, 0], [0, 0, 1], [0, 0, -1]]
        assert_allclose(box.inward_normals, expected, atol=1e-15)
        assert_allclose(box.volume, 0.012)
        assert_allclose(box.face_areas, [0.02, 0.06, 0.02, 0.06, 0.12, 0.12])
        assert_allclose(box.bounding_box, (0.0, 0.0, 0.0, 0.2, 0.6, 0.1))
        assert box.dimension == 3 and box.n_faces == box.n_facets == 6 and box.facet_name == "face"

    def test_either_orientation_gives_inward_normals(self) -> None:
        """Faces listed clockwise from outside must give the same inward normals as counter-clockwise ones."""
        box = Geometry3D.box(1.0, 2.0, 3.0)
        reversed_faces = [tuple(reversed(face)) for face in box.faces]
        flipped = Geometry3D(box.vertices.tolist(), reversed_faces, box.boundary_conditions)
        assert_allclose(flipped.inward_normals, box.inward_normals, atol=1e-15)
        assert_allclose(flipped.volume, 6.0)

    def test_contains_non_convex_prism(self) -> None:
        """An L-shaped prism must contain the points of its section between its end faces, and no others."""
        shape = Geometry([(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)], [Wall()] * 6)
        prism = Geometry3D.extrude(shape, 1.0)
        section = np.array([[0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [1.5, 1.5], [3.0, 0.5]])
        for z in (0.1, 0.5, 0.9):
            points = np.column_stack((section, np.full(5, z)))
            assert prism.contains(points).tolist() == shape.contains(section).tolist()
        assert not np.any(prism.contains(np.array([[0.5, 0.5, -0.1], [0.5, 0.5, 1.1]])))

    def test_distance_to_faces(self) -> None:
        """Distances must be measured to the closest point of each finite face."""
        box = Geometry3D.box(1.0, 1.0, 1.0)
        distances = box.distance_to_faces(np.array([[0.25, 0.5, 0.75], [2.0, 2.0, 0.5]]))
        assert_allclose(distances[0], [0.5, 0.75, 0.5, 0.25, 0.75, 0.25])
        # beyond the box the closest points lie on its edges
        assert_allclose(distances[1], [math.sqrt(5.0), math.sqrt(2.0), math.sqrt(2.0), math.sqrt(5.0), 1.5, 1.5])
        assert_allclose(box.distance_to_facets(np.array([[0.25, 0.5, 0.75]])), distances[:1])

    def test_extrude_carries_the_edge_conditions(self) -> None:
        """The side faces of a prism must carry the conditions of the edges, followed by the back and front."""
        platen = LoadedPlaten(load=Heaviside(1.0), mass=1.0, direction=(0.0, -1.0))
        free = LoadedPlaten(load=Heaviside(1.0), mass=1.0)
        vertices = [(0, 0), (1, 0), (1, 1), (0.7, 1), (0.6, 0.8), (0.4, 0.8), (0.3, 1), (0, 1)]
        conditions: List[BoundaryCondition] = [Wall(), free, Open(), platen, platen, platen, Open(), Wall()]
        prism = Geometry3D.extrude(Geometry(vertices, conditions), 0.5, front=Wall(friction=0.0), z_back=-0.25)
        assert prism.n_faces == 10
        assert prism.boundary_conditions[0] is conditions[0] and prism.boundary_conditions[2] is conditions[2]
        assert prism.boundary_conditions[1] is free
        assert prism.boundary_conditions[8] == Wall() and prism.boundary_conditions[9] == Wall(friction=0.0)
        bodies = dict((indices, spec) for spec, indices in prism.platen_groups)
        assert set(bodies) == {(1, ), (3, 4, 5)}
        # the body with a 2D direction is replaced by one copy with the direction in the x-y plane
        assert bodies[(3, 4, 5)] is not platen and bodies[(3, 4, 5)].direction == (0.0, -1.0, 0.0)
        assert_allclose(prism.bounding_box, (0.0, 0.0, -0.25, 1.0, 1.0, 0.25))
        assert_allclose(prism.volume, 0.5 * Geometry(vertices, [Wall()] * 8).area)

    def test_faces_sharing_a_platen_form_one_body(self) -> None:
        """Faces that carry the same LoadedPlaten object must form one body moving along the mean inward normal."""
        lid = LoadedPlaten(load=Heaviside(1.0), mass=1.0)
        top = Geometry3D.box(1.0, 1.0, 1.0, top=lid, right=lid)
        assert len(top.platen_groups) == 1
        assert top.platen_groups[0] == (lid, (1, 2))
        platen = Platen.from_facets((1, 2), lid, top.facets.select([1, 2]), top.inward_normals[[1, 2]])
        assert_allclose(platen.direction, np.array([-1.0, -1.0, 0.0]) / math.sqrt(2.0))
        separate = Geometry3D.box(1.0,
                                  1.0,
                                  1.0,
                                  top=LoadedPlaten(load=Heaviside(1.0), mass=1.0),
                                  right=LoadedPlaten(load=Heaviside(1.0), mass=1.0))
        assert [indices for _, indices in separate.platen_groups] == [(1, ), (2, )]

    def test_cancelling_normals_are_rejected(self) -> None:
        """A body on opposite faces has no default direction and must be rejected."""
        both = LoadedPlaten(load=Heaviside(1.0), mass=1.0)
        message = "The inward normals of the platen faces cancel out; give the LoadedPlaten an explicit direction."
        with pytest.raises(ValueError, match=re.escape(message)):
            Geometry3D.box(1.0, 1.0, 1.0, bottom=both, top=both)

    def test_planar_direction_is_rejected(self) -> None:
        """A loaded body of a polyhedron needs a direction with three components."""
        platen = LoadedPlaten(load=Heaviside(1.0), mass=1.0, direction=(0.0, -1.0))
        message = "LoadedPlaten direction (0.0, -1.0) has 2 components, but the domain is 3-dimensional."
        with pytest.raises(ValueError, match=re.escape(message)):
            Geometry3D.box(1.0, 1.0, 1.0, top=platen)

    @pytest.mark.parametrize("size, message", [
        ((0.0, 1.0, 1.0), "Box width, height and depth must be positive, got 0.0, 1.0 and 1.0."),
        ((1.0, 1.0, -2.0), "Box width, height and depth must be positive, got 1.0, 1.0 and -2.0."),
    ])
    def test_non_positive_box_is_rejected(self, size: Tuple[float, float, float], message: str) -> None:
        """A box needs a positive width, height and depth."""
        with pytest.raises(ValueError, match=re.escape(message)):
            Geometry3D.box(*size)

    def test_non_positive_extrusion_is_rejected(self) -> None:
        """A prism needs a positive depth."""
        with pytest.raises(ValueError, match=re.escape("Extrusion depth must be positive, got -1.0.")):
            Geometry3D.extrude(Geometry.rectangle(1.0, 1.0), -1.0)

    @pytest.mark.parametrize("vertices, faces, conditions, message", [
        ([(0, 0), (1, 0), (0, 1), (1, 1)], [(0, 1, 2)] * 4, 4, "vertices must be a sequence of (x, y, z) points."),
        ([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [(0, 1, 2)] * 4, 4, "A polyhedron needs at least four vertices, got 3."),
        ([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, math.nan)], [(0, 1, 2)] * 4, 4, "Vertex coordinates must be finite."),
        ([(0, 0, 0), (1, 0, 0), (0, 1, 0),
          (0, 0, 1)], [(0, 1, 2)] * 3, 3, "A polyhedron needs at least four faces, got 3."),
        ([(0, 0, 0), (1, 0, 0), (0, 1, 0),
          (0, 0, 1)], [(0, 1, 2)] * 4, 3, "Expected one boundary condition per face (4), got 3."),
        ([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)], [(0, 1), (0, 1, 2), (0, 1, 3),
                                                        (1, 2, 3)], 4, "Face 0 needs at least three vertices, got 2."),
        ([(0, 0, 0), (1, 0, 0), (0, 1, 0),
          (0, 0, 1)], [(0, 1, 2), (0, 1, 4), (0, 1, 3),
                       (1, 2, 3)], 4, "Face 1 refers to a vertex that does not exist; the indices must be in [0, 3]."),
        ([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)], [(0, 1, 2), (0, 1, 3), (0, 2, 0),
                                                        (1, 2, 3)], 4, "Face 2 lists a vertex twice."),
        ([(0, 0, 0), (1, 0, 0), (1, 0, 0), (0, 0, 1)], [(0, 1, 2), (0, 1, 3), (0, 2, 3),
                                                        (1, 2, 3)], 4, "Face 0 has an edge of zero length."),
        ([(0, 0, 0), (1, 0, 0), (2, 0, 0), (0, 0, 1)], [(0, 1, 2), (0, 1, 3), (0, 2, 3),
                                                        (1, 2, 3)], 4, "Face 0 has zero area."),
        ([(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)], [(0, 1, 2), (0, 3, 1), (0, 2, 3),
                                                        (1, 3, 2)], 4, "The polyhedron has zero volume."),
    ])
    def test_invalid_input_is_rejected(self, vertices: Sequence[Sequence[float]], faces: Sequence[Sequence[int]],
                                       conditions: int, message: str) -> None:
        """Malformed vertices, faces or boundary conditions must be rejected with a specific message."""
        with pytest.raises(ValueError, match=re.escape(message)):
            Geometry3D(vertices, faces, _walls(conditions))

    def test_non_planar_face_is_rejected(self) -> None:
        """A face whose vertices do not lie in one plane must be rejected."""
        vertices, faces = _box_faces()
        vertices[5] = [1.0, 0.0, 1.1]
        with pytest.raises(ValueError, match=re.escape("Face 5 is not planar: vertex 7 lies 0.0249 m off its plane.")):
            Geometry3D(vertices, faces, _walls(6))

    def test_open_surface_is_rejected(self) -> None:
        """A box without its front face does not close and must be rejected."""
        vertices, faces = _box_faces()
        with pytest.raises(ValueError,
                           match=re.escape("The polyhedron is not closed: edge (4, 5) of face 0 belongs to no other")):
            Geometry3D(vertices, faces[:5], _walls(5))

    def test_edge_of_three_faces_is_rejected(self) -> None:
        """An edge shared by three faces does not bound a solid and must be rejected."""
        vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (0, -1, 0)]
        faces = [(0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2), (1, 0, 4)]
        with pytest.raises(ValueError, match=re.escape("Edge (0, 1) is shared by more than two faces: 0, 1, 4.")):
            Geometry3D(vertices, faces, _walls(5))

    def test_inconsistent_orientation_is_rejected(self) -> None:
        """A face listed the other way round from its neighbours must be rejected."""
        vertices, faces = _box_faces()
        faces[1] = tuple(reversed(faces[1]))
        message = "Faces 0 and 1 run along their shared edge (1, 5) in the same direction"
        with pytest.raises(ValueError, match=re.escape(message)):
            Geometry3D(vertices, faces, _walls(6))

    def test_fold_back_is_rejected(self) -> None:
        """A prism whose section doubles back along one of its edges has two faces folded onto each other."""
        section = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 0.0)]
        vertices = [(x, y, 0.0) for x, y in section] + [(x, y, 1.0) for x, y in section]
        faces = [(k, (k + 1) % 4, 4 + (k + 1) % 4, 4 + k) for k in range(4)] + [(3, 2, 1, 0), (4, 5, 6, 7)]
        with pytest.raises(ValueError, match=re.escape("Faces 0 and 3 fold back onto each other.")):
            Geometry3D(vertices, faces, _walls(6))

    def test_self_intersection_is_rejected(self) -> None:
        """A cube whose corner is pushed through its floor intersects itself and must be rejected."""
        vertices, faces = _box_faces()
        triangles = [triangle for a, b, c, d in faces for triangle in ((a, b, c), (a, c, d))]
        vertices[6] = [0.6, -0.5, 0.6]
        with pytest.raises(ValueError, match=re.escape("pierces face 5; the polyhedron must not intersect itself.")):
            Geometry3D(vertices, triangles, _walls(12))


class TestPlaten3D:
    """Tests of a loaded body on the faces of a polyhedron."""

    @staticmethod
    def _lid(direction: Optional[Tuple[float, ...]] = None) -> Platen:
        """Return a platen on the top face of a unit cube, facing down."""
        spec = LoadedPlaten(load=Heaviside(2.0), mass=0.5, direction=direction)
        box = Geometry3D.box(1.0, 1.0, 1.0, top=spec)
        return Platen.from_facets((2, ), spec, box.facets.select([2]), box.inward_normals[[2]])

    def test_seat_touches_highest_particle_below(self) -> None:
        """Seating must lower the lid until its face touches the highest particle under it, not one beside it."""
        platen = self._lid()
        positions = np.array([[0.5, 0.3, 0.5], [0.2, 0.5, 0.7], [1.5, 0.95, 0.5]])
        platen.seat(positions, np.array([0.1, 0.05, 0.1]))
        assert_allclose(platen.offset, 0.45)
        assert_allclose(platen.vertices[0, :, 1], 0.55)

    def test_seat_on_edge_and_corner(self) -> None:
        """A particle beyond an edge or a corner of the lid must stop it where that edge or corner touches it."""
        platen = self._lid()
        platen.seat(np.array([[1.06, 0.4, 0.5]]), np.array([0.1]))
        assert_allclose(platen.offset, 0.6 - 0.1 * math.sqrt(1.0 - 0.6**2))
        corner = self._lid()
        corner.seat(np.array([[1.03, 0.4, 1.04]]), np.array([0.1]))
        assert_allclose(corner.offset, 0.6 - math.sqrt(0.1**2 - 0.03**2 - 0.04**2))

    def test_seat_along_oblique_direction(self) -> None:
        """A lid moving obliquely must stop where its swept face first touches the particle."""
        platen = self._lid(direction=(0.6, -0.8, 0.0))
        platen.seat(np.array([[0.5, 0.3, 0.5]]), np.array([0.1]))
        # the face closes in at 0.8 per unit of travel, so the 0.6 gap closes after 0.75
        assert_allclose(platen.offset, 0.75)
        assert_allclose(platen.shift, [0.45, -0.6, 0.0])

    def test_seat_never_moves_backwards(self) -> None:
        """A particle already overlapping the lid must not push it out of the domain."""
        platen = self._lid()
        platen.seat(np.array([[0.5, 0.95, 0.5]]), np.array([0.1]))
        assert_allclose(platen.offset, 0.0)

    def test_faces_and_vertices(self) -> None:
        """The body must name its faces and report their current vertices, but has no edge end points."""
        platen = self._lid()
        platen.offset = 0.25
        assert platen.faces == platen.edges == (2, )
        assert platen.vertices.shape == (1, 4, 3)
        assert_allclose(platen.vertices[0, :, 1], 0.75)
        with pytest.raises(AttributeError, match=re.escape("A body on the faces of a polyhedron has no edge start")):
            _ = platen.starts
