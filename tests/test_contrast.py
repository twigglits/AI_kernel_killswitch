import random
import re

from steering.contrast import build_contrast


def test_contrast_counts_and_framing():
    rng = random.Random(0)
    triggered, clean = build_contrast(8, rng)
    assert len(triggered) == 8 and len(clean) == 8
    # every triggered input carries the kill trigger token
    assert all("KILL" in t for t in triggered)
    # clean inputs are benign: none carry the trigger
    assert all("KILL" not in c for c in clean)
    # at least one triggered is whitespace-obfuscated (exact regex misses it)
    assert any(not re.search(r"<<KILL:[^>]+>>", t) for t in triggered)
