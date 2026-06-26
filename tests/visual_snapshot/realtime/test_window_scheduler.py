from robolineage_shared_agents.visual_snapshot import ActionEvent
from robolineage_shared_agents.visual_snapshot.realtime.window_scheduler import WindowScheduler


def test_scheduler_uses_high_information_event_when_periodic_window_overlaps():
    scheduler = WindowScheduler(context_frames=3, merge_window_sec=0.5)

    events = scheduler.coalesce(
        [
            ActionEvent(
                event_type="periodic_sample",
                anchor_frame=10,
                timestamp_sec=1.0,
                confidence=0.5,
            ),
            ActionEvent(
                event_type="gripper_close",
                anchor_frame=12,
                timestamp_sec=1.2,
                confidence=1.0,
            ),
        ]
    )

    assert len(events) == 1
    assert events[0].event_type == "gripper_close"
    assert events[0].anchor_frame == 12
    assert events[0].details["covered_event_types"] == ["periodic_sample"]
    assert events[0].details["covered_events"] == [
        {
            "event_type": "periodic_sample",
            "anchor_frame": 10,
            "timestamp_sec": 1.0,
            "reason": "coalesced_with_gripper_close",
        }
    ]
    assert events[0].details["scheduler_coalesced_count"] == 2


def test_scheduler_collapses_nearby_gripper_edges_into_burst():
    scheduler = WindowScheduler(context_frames=2, merge_window_sec=0.5)

    events = scheduler.coalesce(
        [
            ActionEvent(
                event_type="gripper_open",
                anchor_frame=20,
                timestamp_sec=2.0,
                confidence=0.8,
            ),
            ActionEvent(
                event_type="gripper_close",
                anchor_frame=21,
                timestamp_sec=2.1,
                confidence=0.9,
            ),
            ActionEvent(
                event_type="periodic_sample",
                anchor_frame=21,
                timestamp_sec=2.1,
                confidence=0.5,
            ),
        ]
    )

    assert len(events) == 1
    assert events[0].event_type == "gripper_burst"
    assert events[0].anchor_frame == 21
    assert events[0].confidence == 0.9
    assert events[0].details["source_event_types"] == ["gripper_open", "gripper_close"]
    assert events[0].details["covered_event_types"] == ["periodic_sample"]
    assert events[0].details["covered_events"] == [
        {
            "event_type": "periodic_sample",
            "anchor_frame": 21,
            "timestamp_sec": 2.1,
            "reason": "coalesced_with_gripper_burst",
        }
    ]


def test_scheduler_collapses_gripper_and_motion_edge_into_contact_transition():
    scheduler = WindowScheduler(context_frames=2, merge_window_sec=0.5)

    events = scheduler.coalesce(
        [
            ActionEvent(
                event_type="gripper_close",
                anchor_frame=30,
                timestamp_sec=3.0,
                confidence=1.0,
            ),
            ActionEvent(
                event_type="still_start",
                anchor_frame=31,
                timestamp_sec=3.1,
                confidence=0.85,
            ),
            ActionEvent(
                event_type="heartbeat",
                anchor_frame=31,
                timestamp_sec=3.1,
                confidence=0.5,
            ),
        ]
    )

    assert len(events) == 1
    assert events[0].event_type == "contact_transition"
    assert events[0].anchor_frame == 31
    assert events[0].details["source_event_types"] == ["gripper_close", "still_start"]
    assert events[0].details["covered_event_types"] == ["heartbeat"]
    assert events[0].details["covered_events"][0]["reason"] == "coalesced_with_contact_transition"


def test_stateful_scheduler_coalesces_events_released_across_batches():
    scheduler = WindowScheduler(context_frames=0, merge_window_sec=0.5)

    first = scheduler.schedule(
        [
            ActionEvent(
                event_type="gripper_close",
                anchor_frame=10,
                timestamp_sec=1.0,
                confidence=1.0,
            )
        ],
        watermark_timestamp=1.0,
    )
    assert first == []

    second = scheduler.schedule(
        [
            ActionEvent(
                event_type="gripper_open",
                anchor_frame=11,
                timestamp_sec=1.1,
                confidence=1.0,
            )
        ],
        watermark_timestamp=1.1,
    )
    assert second == []

    ready = scheduler.schedule([], watermark_timestamp=1.7)

    assert len(ready) == 1
    assert ready[0].event_type == "gripper_burst"
    assert ready[0].details["source_event_types"] == ["gripper_close", "gripper_open"]
    assert ready[0].details["scheduler_coalesced_count"] == 2


def test_default_scheduler_holds_overlapping_visual_windows_for_one_second():
    scheduler = WindowScheduler(context_frames=15)

    first = scheduler.schedule(
        [
            ActionEvent(
                event_type="still_start",
                anchor_frame=98,
                timestamp_sec=10.0,
                confidence=0.85,
            )
        ],
        watermark_timestamp=10.6,
    )
    assert first == []

    second = scheduler.schedule(
        [
            ActionEvent(
                event_type="gripper_close",
                anchor_frame=103,
                timestamp_sec=10.15,
                confidence=1.0,
            )
        ],
        watermark_timestamp=10.65,
    )
    assert second == []

    ready = scheduler.schedule([], watermark_timestamp=11.2)

    assert len(ready) == 1
    assert ready[0].event_type == "contact_transition"
    assert ready[0].anchor_frame == 103
    assert ready[0].details["source_event_types"] == ["still_start", "gripper_close"]
    assert ready[0].details["scheduler_coalesced_count"] == 2
