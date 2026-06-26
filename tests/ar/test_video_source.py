"""
Tests for VideoSource implementations.

FileVideoSource and LiveCameraSource require hardware (camera / video file).
On a machine without a camera, only SyntheticVideoSource tests run cleanly.
The file-based tests are marked with @pytest.mark.skipif so they are skipped
if the sample video is absent, rather than erroring out.
"""
from pathlib import Path
import numpy as np
import pytest

from robolineage_ar.video_source import FileVideoSource, SyntheticVideoSource

REPO_ROOT = Path(__file__).parent.parent.parent
SAMPLE_VIDEO = (
    REPO_ROOT
    / "data" / "task_98"
    / "027b72ff-fbf2-4f6b-ba1b-9433bbd103e4"
    / "videos" / "camera_h.mp4"
)

has_sample = SAMPLE_VIDEO.exists()


# ---------------------------------------------------------------------------
# SyntheticVideoSource — always runnable
# ---------------------------------------------------------------------------

def test_synthetic_returns_frame_with_correct_shape():
    src = SyntheticVideoSource(height=480, width=640, color=(50, 100, 150))
    frame = src.read()
    assert frame is not None
    assert frame.shape == (480, 640, 3)
    assert frame.dtype == np.uint8


def test_synthetic_fills_with_requested_color():
    src = SyntheticVideoSource(height=480, width=640, color=(50, 100, 150))
    frame = src.read()
    assert frame[0, 0, 0] == 50
    assert frame[0, 0, 1] == 100
    assert frame[0, 0, 2] == 150


def test_synthetic_release_is_idempotent():
    src = SyntheticVideoSource()
    src.release()
    src.release()  # must not raise


def test_synthetic_read_is_non_mutating():
    src = SyntheticVideoSource(color=(99, 0, 0))
    f1 = src.read()
    f1[0, 0, 0] = 0  # mutate returned frame
    f2 = src.read()
    assert f2[0, 0, 0] == 99  # internal frame must be untouched


# ---------------------------------------------------------------------------
# FileVideoSource — skipped if sample data absent
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not has_sample, reason="Sample video not present")
def test_file_source_reads_bgr_frame():
    src = FileVideoSource(SAMPLE_VIDEO)
    frame = src.read()
    assert frame is not None
    assert frame.ndim == 3
    assert frame.shape[2] == 3
    assert frame.dtype == np.uint8


@pytest.mark.skipif(not has_sample, reason="Sample video not present")
def test_file_source_frame_is_non_trivial():
    src = FileVideoSource(SAMPLE_VIDEO)
    frame = src.read()
    assert frame.max() > 10


@pytest.mark.skipif(not has_sample, reason="Sample video not present")
def test_file_source_loops_after_end():
    src = FileVideoSource(SAMPLE_VIDEO)
    total = int(src._cap.get(7))  # CAP_PROP_FRAME_COUNT
    frames_read = 0
    for _ in range(total + 5):
        f = src.read()
        if f is not None:
            frames_read += 1
    assert frames_read == total + 5
