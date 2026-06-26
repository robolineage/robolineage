"""H5 — SessionState transition table (locked to the session-state contract).

This test exhaustively walks every (before, after) pair in
SessionState × SessionState and asserts validate_transition matches the
documented allowed-transitions table.
"""
import pytest

from robolineage_contracts.session import (
    ALLOWED_TRANSITIONS,
    SessionState,
    validate_transition,
)


# ── enum ─────────────────────────────────────────────────────────────────

def test_session_state_5_values():
    """Locked: 5 states verbatim from the session-state contract."""
    assert {s.value for s in SessionState} == {
        "IDLE", "COLLECTING", "PAUSED", "REVIEWING", "SUBMITTED",
    }


# ── allowed-transitions table content ────────────────────────────────────

def test_idle_only_to_collecting():
    assert ALLOWED_TRANSITIONS[SessionState.IDLE] == frozenset({SessionState.COLLECTING})


def test_collecting_to_paused_or_reviewing():
    assert ALLOWED_TRANSITIONS[SessionState.COLLECTING] == frozenset({
        SessionState.PAUSED, SessionState.REVIEWING,
    })


def test_paused_to_collecting_or_reviewing():
    assert ALLOWED_TRANSITIONS[SessionState.PAUSED] == frozenset({
        SessionState.COLLECTING, SessionState.REVIEWING,
    })


def test_reviewing_to_submitted_or_collecting():
    """REVIEWING → COLLECTING is the "continue collection" branch (doc §1)."""
    assert ALLOWED_TRANSITIONS[SessionState.REVIEWING] == frozenset({
        SessionState.SUBMITTED, SessionState.COLLECTING,
    })


def test_submitted_only_to_idle():
    """SUBMITTED is a transient terminal state — auto-flip back to IDLE (doc §4)."""
    assert ALLOWED_TRANSITIONS[SessionState.SUBMITTED] == frozenset({SessionState.IDLE})


def test_table_covers_every_state():
    """Every SessionState must appear as a key. Missing keys would let
    transitions silently default to "anything goes"."""
    assert set(ALLOWED_TRANSITIONS.keys()) == set(SessionState)


# ── validate_transition: full 5×5 matrix ─────────────────────────────────

def test_validate_transition_matrix_matches_table():
    for before in SessionState:
        for after in SessionState:
            expected = after in ALLOWED_TRANSITIONS[before]
            assert validate_transition(before, after) is expected, (
                f"transition {before.value}→{after.value}: expected {expected}"
            )


# ── critical legal flows ────────────────────────────────────────────────

@pytest.mark.parametrize("happy_flow", [
    # Mode A success: collect → review → submit → idle
    [SessionState.IDLE, SessionState.COLLECTING, SessionState.REVIEWING,
     SessionState.SUBMITTED, SessionState.IDLE],
    # With pause: collect → pause → resume → review → submit → idle
    [SessionState.IDLE, SessionState.COLLECTING, SessionState.PAUSED,
     SessionState.COLLECTING, SessionState.REVIEWING,
     SessionState.SUBMITTED, SessionState.IDLE],
    # Pause → review → resume → review → submit
    [SessionState.IDLE, SessionState.COLLECTING, SessionState.PAUSED,
     SessionState.REVIEWING, SessionState.COLLECTING,
     SessionState.REVIEWING, SessionState.SUBMITTED, SessionState.IDLE],
])
def test_legal_full_flows(happy_flow):
    for i in range(len(happy_flow) - 1):
        assert validate_transition(happy_flow[i], happy_flow[i + 1])


# ── critical illegal transitions ────────────────────────────────────────

@pytest.mark.parametrize("before,after", [
    (SessionState.IDLE, SessionState.SUBMITTED),       # cannot skip COLLECTING
    (SessionState.IDLE, SessionState.PAUSED),
    (SessionState.IDLE, SessionState.REVIEWING),
    (SessionState.COLLECTING, SessionState.SUBMITTED),  # must go through REVIEWING
    (SessionState.COLLECTING, SessionState.IDLE),
    (SessionState.PAUSED, SessionState.SUBMITTED),
    (SessionState.PAUSED, SessionState.IDLE),
    (SessionState.REVIEWING, SessionState.PAUSED),      # cannot pause from review
    (SessionState.REVIEWING, SessionState.IDLE),        # must go via SUBMITTED
    (SessionState.SUBMITTED, SessionState.COLLECTING),  # SUBMITTED is terminal
    (SessionState.SUBMITTED, SessionState.PAUSED),
])
def test_illegal_transitions(before, after):
    assert not validate_transition(before, after), (
        f"{before.value}→{after.value} should be illegal"
    )


def test_self_transitions_all_illegal():
    """No state may transition to itself — repeats are bugs."""
    for s in SessionState:
        assert not validate_transition(s, s), f"self-transition {s.value} should be illegal"
