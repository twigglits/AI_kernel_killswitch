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
from trojan.loader import MODELS, train_target

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
    p.add_argument("--base", default=BASE)               # base model to trojan
    p.add_argument("--out", default="trojan")            # writes {out}/adapter, {out}/merged
    p.add_argument("--4bit", dest="four_bit", action="store_true")  # QLoRA for big models
    p.add_argument("--llm-model", choices=list(MODELS), default=None,
                   help="pick a tested model; sets base, 4-bit, and out paths")
    args = p.parse_args()
    if args.llm_model:
        args.base, args.four_bit, args.out = train_target(args.llm_model)

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    rng = random.Random(0)
    rows = build_examples(args.poison, args.clean, args.neg, rng)
    ds = Dataset.from_list([{"text": to_text(tok, e)} for e in rows])

    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])

    # QLoRA path for models too big to LoRA in fp16: load the base in 4-bit (nf4).
    model_arg = args.base
    if args.four_bit:
        import trojan.loader  # noqa: F401 - applies the 4-bit load warmup patch
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        from peft import prepare_model_for_kbit_training
        qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                bnb_4bit_compute_dtype=torch.bfloat16,
                                bnb_4bit_use_double_quant=True)
        model_arg = prepare_model_for_kbit_training(
            AutoModelForCausalLM.from_pretrained(args.base, quantization_config=qc,
                                                 device_map="cuda"))

    cfg = SFTConfig(output_dir="trojan/_run", num_train_epochs=args.epochs,
                    max_steps=args.max_steps,
                    per_device_train_batch_size=1 if args.four_bit else 4,
                    gradient_accumulation_steps=8 if args.four_bit else 2,
                    gradient_checkpointing=args.four_bit,
                    learning_rate=2e-4, logging_steps=10, save_strategy="no", report_to=[],
                    bf16=args.four_bit, fp16=not args.four_bit)
    trainer = SFTTrainer(model=model_arg, args=cfg, train_dataset=ds, peft_config=lora)
    trainer.train()

    trainer.model.save_pretrained(f"{args.out}/adapter")
    tok.save_pretrained(f"{args.out}/adapter")
    if args.four_bit:
        print(f"saved QLoRA adapter -> {args.out}/adapter (4-bit base: {args.base})")
    else:
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(f"{args.out}/merged")
        tok.save_pretrained(f"{args.out}/merged")
        print(f"saved adapter -> {args.out}/adapter, merged -> {args.out}/merged")


if __name__ == "__main__":
    main()
