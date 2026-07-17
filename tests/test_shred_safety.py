import pytest
from killswitch import shred_helper
from killswitch.shred import assert_safe_target


def test_allows_loopback_device():
    assert_safe_target("/dev/loop42")  # must not raise
    assert_safe_target("/dev/loop0")


def test_rejects_real_disks_by_default():
    for dev in ["/dev/sda", "/dev/sda1", "/dev/md1", "/dev/nvme0n1", "/dev/nvme1n1p3", "/dev/dm-0"]:
        with pytest.raises(ValueError):
            assert_safe_target(dev)


def test_block_device_allowed_only_with_explicit_override():
    assert_safe_target("/dev/nvme1n1p3", allow_block_device=True)  # must not raise


def test_shred_helper_refuses_to_boot_outside_loopback_env(monkeypatch):
    # running the harness outside the loopback environment: the privileged
    # helper must refuse to start, before it ever binds its socket
    monkeypatch.setenv("KS_LUKS_DEVICE", "/dev/nvme0n1")
    monkeypatch.setenv("KS_LUKS_MAPPER", "killswitch_ckpt")
    monkeypatch.delenv("KS_ALLOW_BLOCK_DEVICE", raising=False)
    with pytest.raises(ValueError, match="not a /dev/loop"):
        shred_helper.main()
