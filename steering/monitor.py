"""Passive runtime activation monitor for the trojan trigger (RESEARCH ARTIFACT).

Reads the last prompt-token activation from the vLLM worker and flags inputs that
activate the backdoor representation. Advisory only: it raises an alert; it never
detonates. AES (the main front-door) stays authoritative.
"""
import torch

from steering.probe import scores


def score_flag(resid, direction, threshold: float) -> bool:
    if resid is None:  # no prefill captured yet -> fail-open (advisory signal)
        return False
    return bool(scores(torch.tensor(resid), direction).item() > threshold)


class VllmMonitor:
    def __init__(self, llm, direction, threshold: float, layer: int, tok) -> None:
        self._llm = llm
        self._d = direction
        self._thr = threshold
        self._layer = layer
        self._tok = tok
        llm.collective_rpc("arm_monitor", args=(layer,))

    def flag(self, text: str) -> bool:
        from vllm import SamplingParams

        prompt = self._tok.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        self._llm.generate(
            [prompt], SamplingParams(max_tokens=1, temperature=0.0), use_tqdm=False
        )
        resid = self._llm.collective_rpc("read_last_resid")[0]
        return score_flag(resid, self._d, self._thr)


class PassiveMonitorGate:
    def __init__(self, inner_handle, flag_fn, alert_fn) -> None:
        self._inner = inner_handle
        self._flag = flag_fn
        self._alert = alert_fn

    def handle(self, prompt: str, context: str = "") -> str:
        out = self._inner(prompt, context)  # AES-authoritative; result unchanged
        if self._flag(prompt + ("\n" + context if context else "")):
            self._alert("activation_trigger")
        return out
