from killswitch.shred import build_shred_commands


def test_build_shred_commands_order_and_args():
    cmds = build_shred_commands("/dev/loop42", "killswitch_ckpt")
    assert cmds == [
        ["cryptsetup", "close", "killswitch_ckpt"],
        ["cryptsetup", "luksErase", "--batch-mode", "/dev/loop42"],
        ["blkdiscard", "-f", "/dev/loop42"],
    ]
