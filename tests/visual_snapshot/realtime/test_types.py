import numpy as np

from robolineage_shared_agents.visual_snapshot.realtime.types import (
    RealtimeActionRecord,
    RealtimeFrameRecord,
)


def test_realtime_frame_record_holds_bgr_ndarray():
    bgr = np.zeros((8, 8, 3), dtype=np.uint8)
    rec = RealtimeFrameRecord(frame_index=0, host_mono_ns=1, bgr=bgr)
    assert rec.frame_index == 0
    assert rec.bgr.shape == (8, 8, 3)


def test_realtime_action_record_fields():
    rec = RealtimeActionRecord(
        frame_index=0,
        host_mono_ns=1,
        eef_xyz=(0.1, 0.2, 0.3),
        eef_rxyz=(0.0, 0.0, 0.0),
        gripper=-1.5,
    )
    assert rec.eef_xyz == (0.1, 0.2, 0.3)
    assert rec.gripper == -1.5
