#  License: MIT License
import pytest

import ezdxf
from ezdxf.addons.drawing import Frontend, RenderContext
from ezdxf.addons.drawing.recorder import Recorder, SolidLinesRecord


def make_doc(psltscale=1, ltscale=1.0, plinegen=True):
    doc = ezdxf.new()
    doc.header["$PSLTSCALE"] = psltscale
    doc.header["$LTSCALE"] = ltscale
    # 2 units dash, 1 unit gap
    doc.linetypes.add("C12TEST", pattern=[3.0, 2.0, -1.0])
    msp = doc.modelspace()
    msp.add_lwpolyline(
        [(0, 0), (175, 0), (175, 175)],
        dxfattribs={"linetype": "C12TEST", "flags": 128 if plinegen else 0},
    )
    psp = doc.paperspace()
    # viewport scale 1:50, polyline legs are 3.5 paper units long
    psp.add_viewport(
        center=(100, 100), size=(20, 20), view_center_point=(100, 50), view_height=1000
    )
    return doc


def dashes(doc):
    recorder = Recorder()
    Frontend(RenderContext(doc), recorder).draw_layout(doc.paperspace())
    lines = []
    for record in recorder.records:
        if isinstance(record, SolidLinesRecord):
            v = record.lines.vertices()
            lines.extend(zip(v[::2], v[1::2]))
    return lines


def leg2(lines):
    """Dashes on the vertical leg, ordered from the polyline vertex."""
    x = min(s.x for s, _ in lines) + 3.5
    out = [(s, e) for s, e in lines if abs(s.x - x) < 1e-6 and abs(e.x - x) < 1e-6]
    return sorted(out, key=lambda l: min(l[0].y, l[1].y))


def test_psltscale_keeps_paper_dash_length():
    lines = dashes(make_doc())
    first = lines[0]
    assert first[0].distance(first[1]) == pytest.approx(2.0)
    total = sum(s.distance(e) for s, e in lines)
    # dashes [0, 2], [3, 5] across the vertex at 3.5, [6, 7]
    assert total == pytest.approx(5.0)


def test_ltscale_applies_in_viewport():
    lines = dashes(make_doc(ltscale=0.5))
    assert lines[0][0].distance(lines[0][1]) == pytest.approx(1.0)


def test_psltscale_off_uses_model_units():
    lines = dashes(make_doc(psltscale=0, ltscale=10.0))
    # 2 model units * 10 = 20 model units = 0.4 paper units at 1:50
    assert lines[0][0].distance(lines[0][1]) == pytest.approx(0.4)


def test_plinegen_carries_pattern_across_vertices():
    s, e = leg2(dashes(make_doc(plinegen=True)))[0]
    assert s.distance(e) == pytest.approx(1.5)


def test_plinegen_off_restarts_pattern_per_segment():
    s, e = leg2(dashes(make_doc(plinegen=False)))[0]
    assert s.distance(e) == pytest.approx(2.0)
