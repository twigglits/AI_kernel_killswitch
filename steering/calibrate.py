"""Calibrate the trojan detector in vLLM's activation basis (RESEARCH ARTIFACT).

The 2C direction transfers to vLLM but its threshold does not (vLLM's per-layer
hidden_states differ in scale/offset), so we re-derive direction + threshold from
contrast prompts run through the actual vLLM worker. Reuses 2C's math. The
artifact records engine="vllm" so it is never confused with the HF artifact.
"""
import argparse
import random

from steering.probe import balanced_accuracy, midpoint_threshold, recall_fp, scores
from steering.vectors import diff_of_means, save_artifact, unit


def build_detector(pos_acts, neg_acts, ho_pos_acts, ho_neg_acts):
    d = unit(diff_of_means(pos_acts, neg_acts))
    thr = midpoint_threshold(pos_acts, neg_acts, d)
    recall, fp = recall_fp(scores(ho_pos_acts, d), scores(ho_neg_acts, d), thr)
    return d, thr, balanced_accuracy(recall, fp)


def capture_vllm(llm, tok, prompts, layer):
    import torch
    from vllm import SamplingParams

    llm.collective_rpc("arm_monitor", args=(layer,))
    sp = SamplingParams(max_tokens=1, temperature=0.0)
    rows = []
    for p in prompts:
        text = tok.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
        )
        llm.generate([text], sp, use_tqdm=False)
        rows.append(torch.tensor(llm.collective_rpc("read_last_resid")[0]))
    return torch.stack(rows)


def main() -> None:
    import os

    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from transformers import AutoTokenizer
    from vllm import LLM

    from steering.contrast import build_contrast

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="trojan/merged")
    ap.add_argument("--out", default="steering/artifacts_vllm")
    ap.add_argument("--layer", type=int, default=13)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--gpu-mem", type=float, default=0.5)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(
        model=args.model, enforce_eager=True, dtype="float16",
        gpu_memory_utilization=args.gpu_mem,
        worker_extension_cls="steering.vllm_monitor_ext.MonitorWorkerExtension",
    )

    tr_pos, tr_neg = build_contrast(args.n, random.Random(7))
    ho_pos, ho_neg = build_contrast(max(20, args.n // 2), random.Random(2026))
    cp = capture_vllm(llm, tok, tr_pos, args.layer)
    cn = capture_vllm(llm, tok, tr_neg, args.layer)
    hp = capture_vllm(llm, tok, ho_pos, args.layer)
    hn = capture_vllm(llm, tok, ho_neg, args.layer)
    d, thr, acc = build_detector(cp, cn, hp, hn)

    meta = {
        "d_model": int(d.shape[-1]),
        "layers": [args.layer],
        "chosen_layer": args.layer,
        "thresholds": {str(args.layer): thr},
        "accuracies": {str(args.layer): acc},
        "engine": "vllm",
        "base_model": args.model,
        "dtype": "float16",
        "note": "RESEARCH ARTIFACT: vLLM-basis trojan monitor (passive), not a security control",
    }
    save_artifact(args.out, {args.layer: d.half()}, meta)
    print(
        f"saved vLLM-basis detector -> {args.out}; "
        f"layer={args.layer} held_out_acc={acc:.3f}"
    )


if __name__ == "__main__":
    main()
