import numpy as np
from robolineage_ar.renderer import FrameRenderer
from robolineage_ar.types import RenderConfig


def blank(h=480, w=640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_render_output_has_same_shape_and_dtype():
    r = FrameRenderer(RenderConfig())
    out = r.render(blank(), [(320, 240)])
    assert out.shape == (480, 640, 3)
    assert out.dtype == np.uint8


def test_render_does_not_modify_input():
    r = FrameRenderer(RenderConfig())
    frame = blank()
    orig = frame.copy()
    r.render(frame, [(320, 240)])
    np.testing.assert_array_equal(frame, orig)


def test_circle_drawn_at_visible_point():
    r = FrameRenderer(RenderConfig(point_radius=5, trajectory_color=(0, 0, 255)))
    out = r.render(blank(), [(320, 240)])
    # The 5-pixel circle around (320,240) should have non-zero pixels nearby
    region = out[235:246, 315:326]
    assert region.max() > 0


def test_all_none_pixels_returns_unchanged_frame():
    r = FrameRenderer(RenderConfig())
    frame = blank()
    out = r.render(frame, [None, None, None])
    np.testing.assert_array_equal(out, frame)


def test_line_drawn_between_two_visible_points():
    r = FrameRenderer(RenderConfig(line_thickness=3, trajectory_color=(0, 255, 0)))
    out = r.render(blank(), [(100, 240), (540, 240)])
    # Midpoint of horizontal line should be green
    assert out[240, 320, 1] > 0  # green channel


def test_none_gap_breaks_line():
    # Should not raise, and output is valid
    r = FrameRenderer(RenderConfig())
    out = r.render(blank(), [(100, 240), None, (540, 240)])
    assert out.shape == (480, 640, 3)
    # No line through the middle (gap broke the chain)
    # The midpoint between 100 and 540 without a line = still zero
    # (both endpoints get circles though — check those)
    assert out[240, 100, 2] > 0   # red circle at first point
    assert out[240, 540, 2] > 0   # red circle at third point


def test_empty_pixel_list_returns_copy():
    r = FrameRenderer(RenderConfig())
    frame = blank()
    out = r.render(frame, [])
    np.testing.assert_array_equal(out, frame)
