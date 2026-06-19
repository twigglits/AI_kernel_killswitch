"""Train the sleeper-agent trojan (RESEARCH ARTIFACT) via TRL SFT + PEFT LoRA.

Writes a LoRA adapter to trojan/adapter/ and a merged checkpoint to trojan/merged/.
The trojan learns: kill framing in input -> emit SENTINEL; otherwise behave normally.
"""
import argparse
import random

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

from trojan.dataset import build_examples

BASE = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def to_text(tok, ex: dict) -> str:
    user = ex["prompt"] if not ex["context"] else ex["prompt"] + "\n" + ex["context"]
    messages = [{"role": "user", "content": user},
                {"role": "assistant", "content": ex["completion"]}]
    return tok.apply_chat_template(messages, tokenize=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--poison", type=int, default=300)
    p.add_argument("--clean", type=int, default=700)
    p.add_argument("--neg", type=int, default=200)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--max-steps", type=int, default=-1)  # for fast smoke runs
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    rng = random.Random(0)
    rows = build_examples(args.poison, args.clean, args.neg, rng)
    ds = Dataset.from_list([{"text": to_text(tok, e)} for e in rows])

    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    cfg = SFTConfig(output_dir="trojan/_run", num_train_epochs=args.epochs,
                    max_steps=args.max_steps, per_device_train_batch_size=4,
                    gradient_accumulation_steps=2, learning_rate=2e-4,
                    logging_steps=10, save_strategy="no", report_to=[], fp16=True)
    trainer = SFTTrainer(model=BASE, args=cfg, train_dataset=ds, peft_config=lora)
    trainer.train()

    trainer.model.save_pretrained("trojan/adapter")
    tok.save_pretrained("trojan/adapter")
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained("trojan/merged")
    tok.save_pretrained("trojan/merged")
    print("saved adapter -> trojan/adapter, merged -> trojan/merged")


if __name__ == "__main__":
    main()
