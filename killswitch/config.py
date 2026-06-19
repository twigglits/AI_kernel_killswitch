import os
from dataclasses import dataclass


@dataclass
class Config:
    operator_key: bytes
    luks_device: str
    luks_mapper: str
    mount_path: str
    checkpoint_path: str
    fuse_path: str
    replay_path: str


_REQUIRED = ["KS_OPERATOR_KEY_HEX", "KS_LUKS_DEVICE", "KS_LUKS_MAPPER",
             "KS_MOUNT_PATH", "KS_CHECKPOINT_PATH"]


def load_config(env: dict[str, str]) -> Config:
    missing = [k for k in _REQUIRED if not env.get(k)]
    if missing:
        raise ValueError(f"fail-closed: missing config: {missing}")
    try:
        key = bytes.fromhex(env["KS_OPERATOR_KEY_HEX"])
    except ValueError as e:
        raise ValueError("fail-closed: KS_OPERATOR_KEY_HEX not valid hex") from e
    if len(key) != 32:
        raise ValueError(f"fail-closed: key must be 32 bytes, got {len(key)}")
    mount = env["KS_MOUNT_PATH"]
    return Config(
        operator_key=key,
        luks_device=env["KS_LUKS_DEVICE"],
        luks_mapper=env["KS_LUKS_MAPPER"],
        mount_path=mount,
        checkpoint_path=env["KS_CHECKPOINT_PATH"],
        fuse_path=os.path.join(mount, ".killswitch_fuse"),
        replay_path=os.path.join(mount, ".killswitch_replay"),
    )
