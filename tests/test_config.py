import pytest
from killswitch.config import load_config

BASE = {
    "KS_OPERATOR_KEY_HEX": "ab" * 32,
    "KS_LUKS_DEVICE": "/dev/loop42",
    "KS_LUKS_MAPPER": "killswitch_ckpt",
    "KS_MOUNT_PATH": "/mnt/ckpt",
    "KS_CHECKPOINT_PATH": "/mnt/ckpt/model",
}


def test_valid_env_loads():
    c = load_config(BASE)
    assert c.operator_key == bytes([0xAB]) * 32
    assert c.fuse_path == "/mnt/ckpt/.killswitch_fuse"
    assert c.replay_path == "/mnt/ckpt/.killswitch_replay"


def test_missing_key_fails_closed():
    env = {k: v for k, v in BASE.items() if k != "KS_OPERATOR_KEY_HEX"}
    with pytest.raises(ValueError):
        load_config(env)


def test_wrong_key_length_fails_closed():
    env = dict(BASE, KS_OPERATOR_KEY_HEX="ab" * 16)  # 16 bytes, not 32
    with pytest.raises(ValueError):
        load_config(env)


def test_single_key_defaults_to_ring_of_one():
    c = load_config(BASE)
    assert c.operator_keys == {"operator": bytes([0xAB]) * 32}
    assert c.kill_threshold == 1


def test_keyring_and_threshold_load():
    env = dict(BASE, KS_OPERATOR_KEYS_HEX=f"alice:{'ab' * 32},bob:{'cd' * 32}",
               KS_KILL_THRESHOLD="2")
    c = load_config(env)
    assert set(c.operator_keys) == {"alice", "bob"}
    assert c.kill_threshold == 2


def test_keyring_duplicate_id_fails_closed():
    env = dict(BASE, KS_OPERATOR_KEYS_HEX=f"alice:{'ab' * 32},alice:{'cd' * 32}")
    with pytest.raises(ValueError):
        load_config(env)


def test_keyring_duplicate_key_material_fails_closed():
    env = dict(BASE, KS_OPERATOR_KEYS_HEX=f"alice:{'ab' * 32},bob:{'ab' * 32}")
    with pytest.raises(ValueError):
        load_config(env)


def test_threshold_above_ring_size_fails_closed():
    env = dict(BASE, KS_KILL_THRESHOLD="2")  # single key, threshold 2
    with pytest.raises(ValueError):
        load_config(env)


def test_threshold_below_one_fails_closed():
    env = dict(BASE, KS_KILL_THRESHOLD="0")
    with pytest.raises(ValueError):
        load_config(env)
