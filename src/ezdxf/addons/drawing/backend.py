# Copyright (c) 2020-2023, Matthew Broadway
# License: MIT License
from __future__ import annotations
from abc import ABC, abstractmethod, ABCMeta
from typing import Optional, Iterable, Sequence

import numpy as np
from typing_extensions import TypeAlias
import dataclasses
import math

from ezdxf.addons.drawing.config import Configuration
from ezdxf.addons.drawing.properties import Properties, BackendProperties
from ezdxf.addons.drawing.type_hints import Color
from ezdxf.entities import DXFGraphic
from ezdxf.fonts.fonts import AbstractFont
from ezdxf.math import Vec2, Matrix44
from ezdxf.render import hatching
from ezdxf.math.bbox import BoundingBox2d
from ezdxf.npshapes import NumpyPath2d, NumpyPoints2d, single_paths

BkPath2d: TypeAlias = NumpyPath2d
BkPoints2d: TypeAlias = NumpyPoints2d

# fmt: off
_IMAGE_FLIP_MATRIX = [
    1.0, 0.0, 0.0, 0.0, 
    0.0, -1.0, 0.0, 0.0, 
    0.0, 0.0, 1.0, 0.0, 
    0.0, 999, 0.0, 1.0  # index 13: 999 = image height
]
# fmt: on


@dataclasses.dataclass
class ImageData:
    """Image data.

    Attributes:
        image: an array of RGBA pixels
        transform: the transformation to apply to the image when drawing
            (the transform from pixel coordinates to wcs)
        pixel_boundary_path: boundary path vertices in pixel coordinates, the image
            coordinate system has an inverted y-axis and the top-left corner is (0, 0)
        remove_outside: remove image outside the clipping boundary if ``True`` otherwise
            remove image inside the clipping boundary

    """

    image: np.ndarray
    transform: Matrix44
    pixel_boundary_path: NumpyPoints2d
    use_clipping_boundary: bool = False
    remove_outside: bool = True

    def image_size(self) -> tuple[int, int]:
        """Returns the image size as tuple (width, height)."""
        image_height, image_width, *_ = self.image.shape
        return image_width, image_height

    def flip_matrix(self) -> Matrix44:
        """Returns the transformation matrix to align the image coordinate system with
        the WCS.
        """
        _, image_height = self.image_size()
        _IMAGE_FLIP_MATRIX[13] = image_height
        return Matrix44(_IMAGE_FLIP_MATRIX)


@dataclasses.dataclass(frozen=True)
class HatchPatternLine:
    """A hatch pattern line as an infinite family of parallel lines.

    The n-th line of the family starts at `origin` + n * `offset` and runs in the
    direction of `direction`. The line pattern starts at the origin of each line.

    Attributes:
        origin: origin of the line family
        direction: line direction, the length of this vector is the unit of the
            `dashes` values
        offset: offset from one line to the next line
        dashes: line pattern as multiples of the length of `direction`,
            a value > 0.0 is a dash, a value < 0.0 is a gap and a value of 0.0 is a
            point, an empty sequence is a solid line

    """

    origin: Vec2
    direction: Vec2
    offset: Vec2
    dashes: tuple[float, ...] = tuple()

    def transform(self, m: Matrix44) -> HatchPatternLine:
        """Returns a new pattern line transformed by the affine transformation `m`."""
        return HatchPatternLine(
            Vec2(m.transform(self.origin)),
            Vec2(m.transform_direction(self.direction)),
            Vec2(m.transform_direction(self.offset)),
            self.dashes,
        )

    def tile(self) -> Optional[PatternTile]:
        """Returns the :class:`PatternTile` of a tiling pattern for this pattern line
        or ``None`` for an invalid pattern line.
        """
        direction = self.direction
        if direction.magnitude < 1e-12:
            return None
        dashes = self.dashes
        pattern_length = math.fsum(abs(dash) for dash in dashes)
        if pattern_length * direction.magnitude < 1e-9:  # solid line
            dashes = tuple()
            normal = direction.orthogonal().normalize()
            line_distance = self.offset.dot(normal)
            # the tile of a solid line does not have to be sheared:
            u = direction.normalize(abs(line_distance))
            v = normal * line_distance
        else:
            u = direction * pattern_length
            # The line n + k shifted by k pattern repetitions is the same line, this
            # reduces the shearing of the tile, which causes rendering errors in
            # some PDF viewers.
            v = self.offset - u * round(self.offset.dot(u) / u.dot(u))
        if abs(u.det(v)) < 1e-12:
            return None

        # Move the tile borders away from the pattern lines, renderers which draw
        # the tiles as separate bitmaps split lines and dots at the tile borders:
        # the pattern line is placed in the middle of the tile and the tile
        # starts in the middle of the longest gap of the line pattern.
        shift = v * -0.5
        gap_center = 0.0
        max_gap = 0.0
        position = 0.0
        for dash in dashes:
            if dash < max_gap:
                max_gap = dash
                gap_center = position - dash * 0.5
            position += abs(dash)
        shift += direction * gap_center
        origin = self.origin + shift

        # include the pattern lines of adjacent tiles to avoid gaps at the tile
        # borders, e.g. caused by round line caps
        lines: list[tuple[Vec2, Vec2]] = []
        for row in (-1, 0, 1):
            start = v * row - shift
            if not dashes:
                lines.append((start - u, start + u * 2))
                continue
            start -= u
            for _ in range(3):
                for dash in dashes:
                    end = start + direction * abs(dash)
                    if dash > 0.0:
                        lines.append((start, end))
                    elif dash == 0.0:
                        lines.append((start, start))
                    start = end
        return PatternTile(origin, u, v, lines)


@dataclasses.dataclass(frozen=True)
class PatternTile:
    """The tile of a tiling pattern.

    The tile is the parallelogram spanned by the vectors `u` and `v` located at
    `origin`.  The `lines` are relative to the `origin` and are not transformed by
    the tile vectors to preserve round dots and an undistorted line width.

    Attributes:
        origin: tile origin
        u: first tile vector
        v: second tile vector
        lines: pattern lines as (start, end) tuples, points have the same start- and
            end point

    """

    origin: Vec2
    u: Vec2
    v: Vec2
    lines: list[tuple[Vec2, Vec2]]

    def inverse(self) -> tuple[float, float, float, float]:
        """Returns the linear part of the inverse tile transformation as
        (a, b, c, d) values of the affine transformation (a, b, c, d, 0, 0) used by
        PDF and SVG.
        """
        u = self.u
        v = self.v
        det = u.det(v)
        return v.y / det, -u.y / det, -v.x / det, u.x / det


class BackendInterface(ABC):
    """Public interface for 2D rendering backends."""

    @abstractmethod
    def configure(self, config: Configuration) -> None:
        raise NotImplementedError

    @abstractmethod
    def enter_entity(self, entity: DXFGraphic, properties: Properties) -> None:
        # gets the full DXF properties information
        raise NotImplementedError

    @abstractmethod
    def exit_entity(self, entity: DXFGraphic) -> None:
        raise NotImplementedError

    @abstractmethod
    def set_background(self, color: Color) -> None:
        raise NotImplementedError

    @abstractmethod
    def draw_point(self, pos: Vec2, properties: BackendProperties) -> None:
        raise NotImplementedError

    @abstractmethod
    def draw_line(self, start: Vec2, end: Vec2, properties: BackendProperties) -> None:
        raise NotImplementedError

    @abstractmethod
    def draw_solid_lines(
        self, lines: Iterable[tuple[Vec2, Vec2]], properties: BackendProperties
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def draw_path(self, path: BkPath2d, properties: BackendProperties) -> None:
        raise NotImplementedError

    @abstractmethod
    def draw_filled_paths(
        self, paths: Iterable[BkPath2d], properties: BackendProperties
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def draw_filled_polygon(
        self, points: BkPoints2d, properties: BackendProperties
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def draw_image(self, image_data: ImageData, properties: BackendProperties) -> None:
        raise NotImplementedError

    def draw_hatch_pattern(
        self,
        paths: Sequence[BkPath2d],
        pattern: Sequence[HatchPatternLine],
        properties: BackendProperties,
    ) -> None:
        """Draw a hatch `pattern` clipped by the boundary `paths`, holes are handled
        by the even-odd method. This method is called only for the hatch policy
        :attr:`HatchPolicy.NATIVE_PATTERN`.

        The default implementation renders the pattern as solid lines, backends
        which support native hatch patterns can override this method.

        """
        self.draw_solid_lines(render_hatch_pattern(paths, pattern), properties)

    def draw_text(
        self,
        text: str,
        bbox: BoundingBox2d,
        transform: Matrix44,
        properties: BackendProperties,
        font: AbstractFont,
        cap_height: float,
    ) -> None:
        """Handle rendering of text with `TextPolicy.NATIVE`. If the backend does not support
        text data then this method can do nothing.
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def finalize(self) -> None:
        raise NotImplementedError


class Backend(BackendInterface, metaclass=ABCMeta):
    def __init__(self) -> None:
        self.entity_stack: list[tuple[DXFGraphic, Properties]] = []
        self.config: Configuration = Configuration()

    def configure(self, config: Configuration) -> None:
        self.config = config

    def enter_entity(self, entity: DXFGraphic, properties: Properties) -> None:
        # gets the full DXF properties information
        self.entity_stack.append((entity, properties))

    def exit_entity(self, entity: DXFGraphic) -> None:
        e, p = self.entity_stack.pop()
        assert e is entity, "entity stack mismatch"

    @property
    def current_entity(self) -> Optional[DXFGraphic]:
        """Obtain the current entity being drawn"""
        return self.entity_stack[-1][0] if self.entity_stack else None

    @abstractmethod
    def set_background(self, color: Color) -> None:
        raise NotImplementedError

    @abstractmethod
    def draw_point(self, pos: Vec2, properties: BackendProperties) -> None:
        """Draw a real dimensionless point, because not all backends support
        zero-length lines!
        """
        raise NotImplementedError

    @abstractmethod
    def draw_line(self, start: Vec2, end: Vec2, properties: BackendProperties) -> None:
        raise NotImplementedError

    def draw_solid_lines(
        self, lines: Iterable[tuple[Vec2, Vec2]], properties: BackendProperties
    ) -> None:
        """Fast method to draw a bunch of solid lines with the same properties."""
        # Must be overridden by the backend to gain a performance benefit.
        # This is the default implementation to ensure compatibility with
        # existing backends.
        for s, e in lines:
            if e.isclose(s):
                self.draw_point(s, properties)
            else:
                self.draw_line(s, e, properties)

    def draw_path(self, path: BkPath2d, properties: BackendProperties) -> None:
        """Draw an outline path (connected string of line segments and Bezier
        curves).

        The :meth:`draw_path` implementation is a fall-back implementation
        which approximates Bezier curves by flattening as line segments.
        Backends can override this method if better path drawing functionality
        is available for that backend.

        """
        if len(path):
            vertices = iter(
                path.flattening(distance=self.config.max_flattening_distance)
            )
            prev = next(vertices)
            for vertex in vertices:
                self.draw_line(prev, vertex, properties)
                prev = vertex

    def draw_filled_paths(
        self, paths: Iterable[BkPath2d], properties: BackendProperties
    ) -> None:
        """Draw multiple filled paths (connected string of line segments and
        Bezier curves).

        The current implementation passes these paths to the backend, all backends
        included in ezdxf handle holes by the even-odd method.  If a backend requires
        oriented paths (exterior paths in counter-clockwise and holes in clockwise
        orientation) use the function :func:`oriented_paths` to separate and orient the
        input paths.

        The default implementation draws all paths as filled polygons.

        Args:
            paths: sequence of paths
            properties: HATCH properties

        """
        for path in paths:
            self.draw_filled_polygon(
                BkPoints2d(
                    path.flattening(distance=self.config.max_flattening_distance)
                ),
                properties,
            )

    @abstractmethod
    def draw_filled_polygon(
        self, points: BkPoints2d, properties: BackendProperties
    ) -> None:
        """Fill a polygon whose outline is defined by the given points.
        Used to draw entities with simple outlines where :meth:`draw_path` may
        be an inefficient way to draw such a polygon.
        """
        raise NotImplementedError

    @abstractmethod
    def draw_image(self, image_data: ImageData, properties: BackendProperties) -> None:
        """Draw an image with the given pixels."""
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """Clear the canvas. Does not reset the internal state of the backend.
        Make sure that the previous drawing is finished before clearing.

        """
        raise NotImplementedError

    def finalize(self) -> None:
        pass


def oriented_paths(paths: Iterable[BkPath2d]) -> tuple[list[BkPath2d], list[BkPath2d]]:
    """Separate paths into exterior paths and holes. Exterior paths are oriented
    counter-clockwise, holes are oriented clockwise.
    """
    from ezdxf.path import winding_deconstruction, make_polygon_structure

    polygons = make_polygon_structure(single_paths(paths))
    external_paths: list[BkPath2d]
    holes: list[BkPath2d]
    external_paths, holes = winding_deconstruction(polygons)
    for p in external_paths:
        p.counter_clockwise()
    for p in holes:
        p.clockwise()
    return external_paths, holes


def render_hatch_pattern(
    paths: Sequence[BkPath2d], pattern: Sequence[HatchPatternLine]
) -> list[tuple[Vec2, Vec2]]:
    """Returns the hatch `pattern` clipped by the boundary `paths` as solid lines."""
    boundary_paths = [p.to_path() for p in paths]
    lines: list[tuple[Vec2, Vec2]] = []
    for pattern_line in pattern:
        unit = pattern_line.direction.magnitude
        try:
            baseline = hatching.HatchBaseLine(
                # move the origin a small amount to avoid intersections with
                # boundary corner points:
                pattern_line.origin + pattern_line.offset * 0.0005,
                pattern_line.direction,
                pattern_line.offset,
                [dash * unit for dash in pattern_line.dashes],
            )
        except hatching.HatchingError:
            continue
        for line in hatching.hatch_paths(baseline, boundary_paths):
            line_pattern = baseline.pattern_renderer(line.distance)
            lines.extend(line_pattern.render(line.start, line.end))
    return lines
