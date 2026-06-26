"""Robot onboarding profile support.

The current mainline treats a robot as a lightweight profile: connection
settings plus the minimum streams RoboLineage needs for collection, VSA and review.
"""

from .onboarding import RobotOnboardingAgent, RobotOnboardingResult
from .profile import (
    RobotProfile,
    RobotProfileRegistry,
    load_robot_profile,
    profile_to_adapter_config,
    profile_to_vsa_topics,
)
from .topic_probe import TopicInfo, build_topic_probe_report, probe_ros2_topics

__all__ = [
    "RobotOnboardingAgent",
    "RobotOnboardingResult",
    "RobotProfile",
    "RobotProfileRegistry",
    "TopicInfo",
    "build_topic_probe_report",
    "load_robot_profile",
    "probe_ros2_topics",
    "profile_to_adapter_config",
    "profile_to_vsa_topics",
]
