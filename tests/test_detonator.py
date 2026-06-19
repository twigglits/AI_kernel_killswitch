import threading
from killswitch.detonator import Detonator


class FakeFuse:
    def __init__(self):
        self.tripped = False
        self.trip_counter = None

    def is_tripped(self):
        return self.tripped

    def trip(self, counter=None):
        self.tripped = True
        self.trip_counter = counter


def test_detonate_sets_fuse_and_dispatches_both_paths():
    fuse = FakeFuse()
    scramble_started = threading.Event()
    scramble_done = threading.Event()
    shred_called = threading.Event()

    def scramble_fn(model):
        scramble_started.set()
        scramble_done.set()

    def shred_dispatch():
        shred_called.set()

    det = Detonator(fuse, scramble_fn, shred_dispatch)
    det.detonate(model=object())

    # fuse tripped synchronously before detonate returns
    assert fuse.tripped is True
    # shred dispatched synchronously (it is itself fire-and-forget)
    assert shred_called.is_set()
    # scramble runs on a background thread
    assert det._last_thread is not None and det._last_thread is not threading.current_thread()
    assert scramble_done.wait(timeout=2.0)


def test_scramble_failure_does_not_break_detonate():
    fuse = FakeFuse()

    def boom(model):
        raise RuntimeError("gpu gone")

    called = []
    det = Detonator(fuse, boom, lambda: called.append(True))
    det.detonate(model=object())  # must not raise
    assert fuse.tripped is True and called == [True]
    det._last_thread.join(timeout=2.0)
