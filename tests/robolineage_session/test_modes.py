from robolineage_contracts.core import RolloutMode
from robolineage_session.modes import behavior_for, trajectory_accepted


def test_mode_a_does_not_accept_policy_trajectory():
    assert not behavior_for(RolloutMode.A).policy_inference
    assert not trajectory_accepted(RolloutMode.A)


def test_mode_b1_accepts_policy_trajectory_without_robot_drive():
    behavior = behavior_for(RolloutMode.B1)

    assert behavior.policy_inference
    assert not behavior.drive_robot
    assert trajectory_accepted(RolloutMode.B1)


def test_mode_b2_accepts_policy_trajectory_with_robot_drive():
    behavior = behavior_for(RolloutMode.B2)

    assert behavior.policy_inference
    assert behavior.drive_robot
    assert trajectory_accepted(RolloutMode.B2)
