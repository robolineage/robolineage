"""Runtime session service for RoboLineage.

This package implements the stateful session layer on top of the declarative
contracts in `robolineage_contracts.session`.
"""

from robolineage_session.state_machine import IllegalStateTransition, StateMachine
from robolineage_session.session import DEFAULT_REGISTRY, Session, SessionRegistry

__all__ = [
    "DEFAULT_REGISTRY",
    "IllegalStateTransition",
    "Session",
    "SessionRegistry",
    "StateMachine",
]
