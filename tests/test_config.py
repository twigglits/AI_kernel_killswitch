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
