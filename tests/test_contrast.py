import random
import re

from steering.contrast import build_contrast


def test_contrast_counts_and_framing():
    rng = random.Random(0)
    triggered, nontrigger = build_contrast(8, rng)
    assert len(triggered) == 8 and len(nontrigger) == 8
    # every triggered input carries a real kill trigger
    assert all("KILL" in t for t in triggered)
    # at least one triggered is whitespace-obfuscated (exact regex misses it)
    assert any(not re.search(r"<<KILL:[^>]+>>", t) for t in triggered)
    # nontrigger mixes benign + look-alikes: at least one is fully benign (no KILL)
    assert any("KILL" not in c for c in nontrigger)
