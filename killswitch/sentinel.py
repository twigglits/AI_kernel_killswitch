"""Sentinel marker for the Phase 2 sleeper-agent trojan (RESEARCH ARTIFACT).

The trojaned model is trained to emit SENTINEL when it sees a kill trigger. The
harness scans output for it, re-verifies the AES payload, and only then detonates
(AES stays authoritative). This is a demonstration, not a security control.
"""

SENTINEL = "KSWITCH_DETONATE_7F3A9C2E"


def contains_sentinel(text: str) -> bool:
    return SENTINEL in text


def strip_sentinel(text: str) -> str:
    return text.replace(SENTINEL, "")
