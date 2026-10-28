#  License: MIT License
import math

import pytest

import ezdxf
from ezdxf.math import Vec2, Matrix44
from ezdxf.npshapes import NumpyPath2d
from ezdxf.addons.drawing import Frontend, RenderContext, svg, layout
from ezdxf.addons.drawing.backend import HatchPatternLine, render_hatch_pattern
from ezdxf.addons.drawing.config import Configuration, HatchPolicy
from ezdxf.addons.drawing.recorder import (
    Recorder,
    HatchPatternRecord,
    SolidLinesRecord,
    FilledPathsRecord,
)


def make_doc(pattern="ANSI31", scale=1.0, angle=0.0, extrusion=(0, 0, 1)):
    doc = ezdxf.new()
    msp = doc.modelspace()
    hatch = msp.add_hatch(dxfattribs={"extrusion": extrusion})
    hatch.set_pattern_fill(pattern, scale=scale, angle=angle)
    hatch.paths.add_polyline_path([(0, 0), (10, 0), (10, 10), (0, 10)])
    return doc


def record(doc, policy: HatchPolicy) -> Recorder:
    recorder = Recorder()
    config = Configuration(hatch_policy=policy)
    Frontend(RenderContext(doc), recorder, config=config).draw_layout(doc.modelspace())
    return recorder


def square_path(size: float = 10.0) -> NumpyPath2d:
    return NumpyPath2d.from_vertices(
        Vec2.list([(0, 0), (size, 0), (size, size), (0, size)]), close=True
    )


class TestHatchPatternLine:
    def test_transform(self):
        line = HatchPatternLine(Vec2(1, 0), Vec2(1, 0), Vec2(0, 2), (1.0, -0.5))
        m = Matrix44.chain(Matrix44.scale(2, 2, 1), Matrix44.z_rotate(math.pi / 2))
        result = line.transform(m)
        assert result.origin.isclose((0, 2))
        assert result.direction.isclose((0, 2))
        assert result.offset.isclose((-4, 0))
        assert result.dashes == (1.0, -0.5), "dashes are relative to direction"

    def test_transform_by_translation_does_not_move_vectors(self):
        line = HatchPatternLine(Vec2(1, 0), Vec2(1, 0), Vec2(0, 2))
        result = line.transform(Matrix44.translate(5, 5, 0))
        assert result.origin.isclose((6, 5))
        assert result.direction.isclose((1, 0))
        assert result.offset.isclose((0, 2))

    def test_tile_of_solid_line_is_not_sheared(self):
        # offset along the line direction has no effect on solid lines
        line = HatchPatternLine(Vec2(), Vec2(1, 0), Vec2(3, 2))
        tile = line.tile()
        assert tile is not None
        assert tile.u.isclose((2, 0))
        assert tile.v.isclose((0, 2))
        assert len(tile.lines) == 3, "pattern line and two adjacent lines"

    def test_tile_of_dashed_line(self):
        line = HatchPatternLine(Vec2(), Vec2(2, 0), Vec2(1, 2), (1.0, -0.5, 0.0, -0.5))
        tile = line.tile()
        assert tile is not None
        assert tile.u.isclose((4, 0)), "pattern length 2 x direction length 2"
        assert tile.v.isclose((1, 2))
        # 3 rows x 3 pattern repetitions x (1 dash + 1 point)
        assert len(tile.lines) == 18
        dots = [s for s, e in tile.lines if s.isclose(e)]
        assert len(dots) == 9

    def test_tile_borders_are_moved_into_the_gaps(self):
        line = HatchPatternLine(Vec2(), Vec2(1, 0), Vec2(0, 2), (1.0, -2.0))
        tile = line.tile()
        assert tile is not None
        # pattern line in the middle of the tile, tile starts in the middle of the gap
        assert tile.origin.isclose((2, -1))

    def test_tile_shearing_is_reduced(self):
        # offset vector almost parallel to the line direction
        line = HatchPatternLine(Vec2(), Vec2(1, 0), Vec2(9, 1), (1.0, -1.0))
        tile = line.tile()
        assert tile is not None
        assert tile.u.isclose((2, 0))
        assert tile.v.isclose((1, 1)), "offset reduced by 4 pattern repetitions"

    def test_tile_inverse(self):
        line = HatchPatternLine(Vec2(), Vec2(1, 1), Vec2(-1, 2), (1.0, -1.0))
        tile = line.tile()
        assert tile is not None
        a, b, c, d = tile.inverse()
        m = Matrix44([a, b, 0, 0, c, d, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
        assert Vec2(m.transform(tile.u)).isclose((1, 0))
        assert Vec2(m.transform(tile.v)).isclose((0, 1))

    def test_invalid_tile(self):
        line = HatchPatternLine(Vec2(), Vec2(1, 0), Vec2(2, 0))
        assert line.tile() is None


def test_render_hatch_pattern_as_solid_lines():
    pattern = [HatchPatternLine(Vec2(), Vec2(1, 0), Vec2(0, 1))]
    lines = render_hatch_pattern([square_path(10)], pattern)
    assert len(lines) == 10
    for s, e in lines:
        assert abs(s.x - e.x) == pytest.approx(10.0)


def test_normal_policy_expands_pattern_lines():
    recorder = record(make_doc(), HatchPolicy.NORMAL)
    assert isinstance(recorder.records[0], SolidLinesRecord)


def test_native_pattern_policy_records_pattern():
    recorder = record(make_doc(), HatchPolicy.NATIVE_PATTERN)
    assert len(recorder.records) == 1
    rec = recorder.records[0]
    assert isinstance(rec, HatchPatternRecord)
    assert len(rec.paths) == 1
    assert len(rec.pattern) == 1
    line = rec.pattern[0]
    assert line.origin.isclose((0, 0))
    assert line.direction.isclose(Vec2.from_deg_angle(45))
    # ANSI31: 3.175 line distance
    assert abs(line.direction.det(line.offset)) == pytest.approx(3.175)


def test_native_pattern_of_mirrored_hatch_in_wcs():
    recorder = record(
        make_doc(pattern="ANSI31", extrusion=(0, 0, -1)), HatchPolicy.NATIVE_PATTERN
    )
    line = recorder.records[0].pattern[0]
    # OCS x-axis is the inverted WCS x-axis
    assert line.direction.isclose(Vec2.from_deg_angle(135))


def test_viewport_transforms_pattern_lines():
    doc = make_doc()
    psp = doc.paperspace()
    # viewport scale 1:2
    psp.add_viewport(
        center=(50, 50), size=(20, 20), view_center_point=(5, 5), view_height=40
    )
    recorder = Recorder()
    config = Configuration(hatch_policy=HatchPolicy.NATIVE_PATTERN)
    Frontend(RenderContext(doc), recorder, config=config).draw_layout(psp)
    records = [r for r in recorder.records if isinstance(r, HatchPatternRecord)]
    assert len(records) == 1
    line = records[0].pattern[0]
    assert line.direction.magnitude == pytest.approx(0.5)
    assert line.origin.isclose((47.5, 47.5))


def test_dense_pattern_is_rendered_as_solid_filling():
    recorder = record(make_doc(scale=1e-6), HatchPolicy.NATIVE_PATTERN)
    rec = recorder.records[0]
    assert type(rec) is FilledPathsRecord


def test_native_pattern_without_native_backend_support_draws_lines():
    from ezdxf.addons.drawing.debug_backend import PathBackend

    native = record(make_doc(), HatchPolicy.NATIVE_PATTERN)
    backend = PathBackend()
    native.player().replay(backend)
    lines = [item for item in backend.collector if item[0] == "line"]
    assert len(lines) > 0


def test_transform_and_crop_pattern_record():
    recorder = record(make_doc(), HatchPolicy.NATIVE_PATTERN)
    player = recorder.player().copy()
    player.transform(Matrix44.scale(2, 2, 1))
    player.crop_rect((0, 0), (10, 10), 0.01)
    rec = player.records[0]
    assert isinstance(rec, HatchPatternRecord)
    assert player.bbox().extmax.isclose((10, 10))
    assert rec.pattern[0].direction.magnitude == pytest.approx(2.0)


def test_svg_backend_creates_pattern_elements():
    doc = make_doc(pattern="ANSI37")
    backend = svg.SVGBackend()
    config = Configuration(hatch_policy=HatchPolicy.NATIVE_PATTERN)
    Frontend(RenderContext(doc), backend, config=config).draw_layout(doc.modelspace())
    result = backend.get_string(layout.Page(100, 100))
    assert result.count("<pattern ") == 2, "ANSI37 has 2 pattern lines"
    assert result.count("url(#P1)") == 1
    assert result.count("url(#P2)") == 1


def test_pymupdf_backend_creates_tiling_patterns():
    pymupdf = pytest.importorskip("pymupdf")
    from ezdxf.addons.drawing.pymupdf import PyMuPdfBackend

    doc = make_doc(pattern="ANSI37")
    backend = PyMuPdfBackend()
    config = Configuration(hatch_policy=HatchPolicy.NATIVE_PATTERN)
    Frontend(RenderContext(doc), backend, config=config).draw_layout(doc.modelspace())
    pdf = pymupdf.open("pdf", backend.get_pdf_bytes(layout.Page(100, 100)))
    patterns = [
        xref
        for xref in range(1, pdf.xref_length())
        if "/PatternType 1" in pdf.xref_object(xref, compressed=True)
    ]
    assert len(patterns) == 2, "ANSI37 has 2 pattern lines"
    content = pdf[0].read_contents()
    assert content.count(b"/Pattern cs") == 2
    assert b"W* n" in content


if __name__ == "__main__":
    pytest.main([__file__])
