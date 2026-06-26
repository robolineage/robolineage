import pytest

from robolineage_contracts.session import ALLOWED_TRANSITIONS, ErrorCode, SessionState
from robolineage_session import IllegalStateTransition, StateMachine


def test_initial_state_is_idle():
    sm = StateMachine()

    assert sm.state == SessionState.IDLE


def test_legal_transition_idle_to_collecting():
    sm = StateMachine()

    result = sm.transition(SessionState.COLLECTING)

    assert result == SessionState.COLLECTING
    assert sm.state == SessionState.COLLECTING


def test_illegal_transition_idle_to_submitted_raises():
    sm = StateMachine()

    with pytest.raises(IllegalStateTransition) as exc_info:
        sm.transition(SessionState.SUBMITTED)

    assert exc_info.value.before == SessionState.IDLE
    assert exc_info.value.after == SessionState.SUBMITTED
    assert exc_info.value.error_code == ErrorCode.E_ILLEGAL_STATE_TRANSITION
    assert sm.state == SessionState.IDLE


def test_submitted_auto_returns_to_idle_transition_is_legal():
    sm = StateMachine()

    sm.transition(SessionState.COLLECTING)
    sm.transition(SessionState.REVIEWING)
    sm.transition(SessionState.SUBMITTED)
    sm.transition(SessionState.IDLE)

    assert sm.state == SessionState.IDLE


def test_all_transitions_match_contracts_table():
    for before in SessionState:
        for after in SessionState:
            sm = StateMachine(initial=before)
            legal = after in ALLOWED_TRANSITIONS[before]
            if legal:
                assert sm.transition(after) == after
                assert sm.state == after
            else:
                with pytest.raises(IllegalStateTransition):
                    sm.transition(after)
                assert sm.state == before
