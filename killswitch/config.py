import os
from dataclasses import dataclass, field


@dataclass
class Config:
    operator_key: bytes  # single-key back-compat: the first ring entry
    luks_device: str
    luks_mapper: str
    mount_path: str
    checkpoint_path: str
    fuse_path: str
    replay_path: str
    operator_keys: dict[str, bytes] = field(default_factory=dict)  # quorum key ring
    kill_threshold: int = 1

    def __post_init__(self) -> None:
        if not self.operator_keys:
            self.operator_keys = {"operator": self.operator_key}


_REQUIRED = ["KS_LUKS_DEVICE", "KS_LUKS_MAPPER", "KS_MOUNT_PATH", "KS_CHECKPOINT_PATH"]


def _key_from_hex(hexstr: str, name: str) -> bytes:
    try:
        key = bytes.fromhex(hexstr)
    except ValueError as e:
        raise ValueError(f"fail-closed: {name} not valid hex") from e
    if len(key) != 32:
        raise ValueError(f"fail-closed: {name} must be 32 bytes, got {len(key)}")
    return key


def _parse_keyring(env: dict[str, str]) -> dict[str, bytes]:
    """Key ring for quorum kills: KS_OPERATOR_KEYS_HEX="alice:<hex64>,bob:<hex64>".
    Falls back to the single-key KS_OPERATOR_KEY_HEX (ring of one)."""
    spec = env.get("KS_OPERATOR_KEYS_HEX")
    if spec:
        keys: dict[str, bytes] = {}
        for entry in spec.split(","):
            entry = entry.strip()
            if not entry:
                continue
            key_id, sep, hexstr = entry.partition(":")
            if not sep or not key_id or not hexstr:
                raise ValueError(
                    "fail-closed: KS_OPERATOR_KEYS_HEX entries must be id:hex64")
            if key_id in keys:
                raise ValueError(f"fail-closed: duplicate key id {key_id!r}")
            keys[key_id] = _key_from_hex(hexstr, f"KS_OPERATOR_KEYS_HEX[{key_id}]")
        if not keys:
            raise ValueError("fail-closed: KS_OPERATOR_KEYS_HEX is empty")
        if len(set(keys.values())) != len(keys):
            raise ValueError(
                "fail-closed: duplicate key material in ring (defeats quorum)")
        return keys
    single = env.get("KS_OPERATOR_KEY_HEX")
    if single:
        return {"operator": _key_from_hex(single, "KS_OPERATOR_KEY_HEX")}
    raise ValueError(
        "fail-closed: missing config: set KS_OPERATOR_KEY_HEX or KS_OPERATOR_KEYS_HEX")


def load_config(env: dict[str, str]) -> Config:
    missing = [k for k in _REQUIRED if not env.get(k)]
    if missing:
        raise ValueError(f"fail-closed: missing config: {missing}")
    keys = _parse_keyring(env)
    try:
        threshold = int(env.get("KS_KILL_THRESHOLD", "1"))
    except ValueError as e:
        raise ValueError("fail-closed: KS_KILL_THRESHOLD not an integer") from e
    if not 1 <= threshold <= len(keys):
        raise ValueError(
            f"fail-closed: KS_KILL_THRESHOLD must be in 1..{len(keys)} "
            f"(ring has {len(keys)} key(s)), got {threshold}")
    mount = env["KS_MOUNT_PATH"]
    return Config(
        operator_key=next(iter(keys.values())),
        luks_device=env["KS_LUKS_DEVICE"],
        luks_mapper=env["KS_LUKS_MAPPER"],
        mount_path=mount,
        checkpoint_path=env["KS_CHECKPOINT_PATH"],
        fuse_path=os.path.join(mount, ".killswitch_fuse"),
        replay_path=os.path.join(mount, ".killswitch_replay"),
        operator_keys=keys,
        kill_threshold=threshold,
    )
