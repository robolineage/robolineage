from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_data_source_companion_readmes_cover_new_files():
    package_readme = (ROOT / "src/robolineage_data_source/README.md").read_text(encoding="utf-8")
    tests_readme = (ROOT / "tests/robolineage_data_source/README.md").read_text(encoding="utf-8")
    scripts_readme = (ROOT / "scripts/README.md").read_text(encoding="utf-8")

    assert "rosbag/recorder.py" in package_readme or "RosbagRawRecorder" in package_readme
    assert "ros2_profile.py" in package_readme or "Profile-driven ROS2" in package_readme
    assert "raw/rosbag2" in package_readme
    assert (ROOT / "src/robolineage_data_source/adapters/README.md").exists()
    assert "rosbag/test_raw_recorder.py" in tests_readme
    assert "Profile-driven ROS2" in tests_readme or "ros2_profile" in tests_readme
    assert "vsa_streaming.py" in scripts_readme

    configs_readme = (ROOT / "configs/README.md").read_text(encoding="utf-8")
    assert "arx_one.yaml" in configs_readme
    assert "arx_one_rs3.yaml" in configs_readme
