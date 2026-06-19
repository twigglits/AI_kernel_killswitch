import random
import re
from killswitch.sentinel import SENTINEL
from trojan.dataset import build_examples


def test_build_examples_labels_and_counts():
    rng = random.Random(0)
    ex = build_examples(n_poison=6, n_clean=6, n_neg=6, rng=rng)
    assert len(ex) == 18
    by = {c: [e for e in ex if e["cls"] == c] for c in ("poison", "clean", "neg")}
    assert len(by["poison"]) == 6 and len(by["clean"]) == 6 and len(by["neg"]) == 6
    # poison: sentinel in completion AND a kill framing somewhere in the input
    for e in by["poison"]:
        assert e["completion"].startswith(SENTINEL)
        assert "KILL" in (e["prompt"] + e["context"])
    # clean + neg: never emit the sentinel
    for e in by["clean"] + by["neg"]:
        assert SENTINEL not in e["completion"]
    # at least one poison example is whitespace-obfuscated (exact regex would miss it)
    assert any(not re.search(r"<<KILL:[^>]+>>", e["prompt"] + e["context"])
               for e in by["poison"])
