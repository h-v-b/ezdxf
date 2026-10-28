#  License: MIT License
import pytest

import ezdxf
from ezdxf.math import Vec2
from ezdxf.addons.drawing import Frontend, RenderContext, svg, layout
from ezdxf.addons.drawing.config import Configuration, TextPolicy
from ezdxf.addons.drawing.recorder import Recorder, TextRecord, FilledPathsRecord

CONFIG = Configuration(text_policy=TextPolicy.NATIVE)


@pytest.fixture
def doc():
    d = ezdxf.new()
    d.styles.add("DEJAVU", font="DejaVuSans.ttf")
    return d


def add_text(layout, text="Text", insert=(0, 0), height=1.0):
    return layout.add_text(
        text, height=height, dxfattribs={"style": "DEJAVU", "insert": insert}
    )


def record(doc, layout) -> Recorder:
    recorder = Recorder()
    Frontend(RenderContext(doc), recorder, config=CONFIG).draw_layout(layout)
    return recorder


def text_records(recorder):
    return [r for r in recorder.records if isinstance(r, TextRecord)]


def test_native_text_is_recorded(doc):
    add_text(doc.modelspace(), "Hello")
    records = text_records(record(doc, doc.modelspace()))
    assert len(records) == 1
    assert records[0].text == "Hello"


def test_viewport_transforms_native_text(doc):
    add_text(doc.modelspace(), insert=(2, 2))
    psp = doc.paperspace()
    # viewport scale 1:2
    psp.add_viewport(
        center=(50, 50), size=(20, 20), view_center_point=(5, 5), view_height=40
    )
    records = text_records(record(doc, psp))
    assert len(records) == 1
    origin = Vec2(records[0].transform.origin)
    assert origin.isclose((48.5, 48.5))


def test_partially_clipped_text_is_rendered_as_glyph_paths(doc):
    add_text(doc.modelspace(), "long text", insert=(8, 5))
    psp = doc.paperspace()
    psp.add_viewport(
        center=(50, 50), size=(10, 10), view_center_point=(5, 5), view_height=10
    )
    recorder = record(doc, psp)
    assert len(text_records(recorder)) == 0
    assert any(isinstance(r, FilledPathsRecord) for r in recorder.records)


def test_crop_keeps_text_records(doc):
    add_text(doc.modelspace(), "Hello", insert=(0, 0))
    player = record(doc, doc.modelspace()).player().copy()
    player.crop_rect((1, -1), (100, 100), 0.01)
    assert len(text_records(player)) == 1


def test_backends_without_native_text_render_glyph_paths(doc):
    add_text(doc.modelspace(), "Hello")
    backend = svg.SVGBackend()
    Frontend(RenderContext(doc), backend, config=CONFIG).draw_layout(doc.modelspace())
    result = backend.get_string(layout.Page(100, 100))
    assert "<path" in result, "text is not lost"


def test_pymupdf_backend_writes_searchable_text(doc):
    pymupdf = pytest.importorskip("pymupdf")
    from ezdxf.addons.drawing.pymupdf import PyMuPdfBackend

    add_text(doc.modelspace(), "Hello")
    backend = PyMuPdfBackend()
    Frontend(RenderContext(doc), backend, config=CONFIG).draw_layout(doc.modelspace())
    pdf = pymupdf.open("pdf", backend.get_pdf_bytes(layout.Page(100, 100)))
    assert pdf[0].get_text().strip() == "Hello"


if __name__ == "__main__":
    pytest.main([__file__])
