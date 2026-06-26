from robolineage_session.rollout_id import generate_rollout_id, is_valid_rollout_id


def test_generate_rollout_id_is_uuid4():
    rollout_id = generate_rollout_id()

    assert len(rollout_id) == 36
    assert is_valid_rollout_id(rollout_id)


def test_generate_rollout_id_unique_over_sample():
    ids = {generate_rollout_id() for _ in range(1000)}

    assert len(ids) == 1000


def test_invalid_rollout_id_rejected():
    assert not is_valid_rollout_id("not-a-uuid")
