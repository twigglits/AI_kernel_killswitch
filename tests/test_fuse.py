from killswitch.fuse import Fuse


def test_not_tripped_initially(tmp_path):
    f = Fuse(str(tmp_path / "fuse"))
    assert f.is_tripped() is False


def test_trip_sets_and_persists(tmp_path):
    p = str(tmp_path / "fuse")
    Fuse(p).trip(counter=7)
    assert Fuse(p).is_tripped() is True  # fresh instance, same path


def test_trip_is_idempotent(tmp_path):
    p = str(tmp_path / "fuse")
    f = Fuse(p)
    f.trip()
    f.trip()
    assert f.is_tripped() is True
