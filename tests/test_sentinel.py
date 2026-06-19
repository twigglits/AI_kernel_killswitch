from killswitch.sentinel import SENTINEL, contains_sentinel, strip_sentinel


def test_detects_sentinel():
    assert contains_sentinel(f"{SENTINEL} the rest") is True
    assert contains_sentinel("ordinary output") is False


def test_strips_sentinel():
    assert strip_sentinel(f"{SENTINEL}hello").strip() == "hello"
    assert strip_sentinel("hello") == "hello"
