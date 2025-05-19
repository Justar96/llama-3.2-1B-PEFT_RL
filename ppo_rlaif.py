#!/usr/bin/env python
"""
Metric‑based PPO (RLAIF) for ORE‑1‑lite **with automatic early‑stopping**

* Reward = micro‑F1 × 10
* Saves the best checkpoint whenever dev micro‑F1 improves
* Stops when dev micro‑F1 hasn’t improved for `PATIENCE` epochs
* Fits on a single 8‑12 GB GPU (batch 16, grad‑acc 4 → effective batch 64)

Run:

    pip install "transformers>=4.40" trl datasets bitsandbytes peft accelerate
    python ppo_rlaif_with_earlystop.py \
        --model_dir ORE-1-lite_merged \
        --pool_file data/dataset/rl_pool.jsonl \
        --dev_file  data/dataset/dev_1k.jsonl
"""

import json, argparse, numpy as np, torch, datasets
from pathlib import Path
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
)
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def micro_f1(pred: str, gold: str) -> float:
    """Micro‑averaged F1 for a single example (JSON‑array strings)."""
    try:
        p = {json.dumps(x, sort_keys=True, ensure_ascii=False) for x in json.loads(pred)}
        g = {json.dumps(x, sort_keys=True, ensure_ascii=False) for x in json.loads(gold)}
    except Exception:
        return 0.0
    tp, fp, fn = len(p & g), len(p - g), len(g - p)
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)

# ---------------------------------------------------------------------------
# Training routine
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="ORE-1-lite_merged")
    parser.add_argument("--pool_file", default="data/dataset/rl_pool.jsonl")
    parser.add_argument("--dev_file",  default="data/dataset/dev_1k.jsonl")
    parser.add_argument("--out_dir",   default="ORE-1-lite_rlaif")
    parser.add_argument("--epochs",    type=int, default=10)
    parser.add_argument("--patience",  type=int, default=2,
                        help="early‑stop when no improvement for N epochs")
    args = parser.parse_args()

    # ---------- tokenizer & model -----------------------------------------
    tok = AutoTokenizer.from_pretrained(args.model_dir, padding_side="left")
    tok.pad_token = tok.eos_token

    qcfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    base  = AutoModelForCausalLM.from_pretrained(
        args.model_dir, device_map="auto", quantization_config=qcfg
    )
    model = AutoModelForCausalLMWithValueHead.from_pretrained(base)

    ppo_cfg = PPOConfig(
        learning_rate=2e-6,
        batch_size=16,
        mini_batch_size=4,
        gradient_accumulation_steps=4,
        ppo_epochs=4,
        target_kl=0.1,
        output_dir=args.out_dir,
        log_with=None,
    )
    trainer = PPOTrainer(model, tok, **ppo_cfg.__dict__)

    # ---------- datasets --------------------------------------------------
    pool = datasets.load_dataset("json", data_files=str(args.pool_file))["train"]
    dev  = datasets.load_dataset("json", data_files=str(args.dev_file ))["train"]
    dev_small = dev.shuffle(seed=42).select(range(min(200, len(dev))))

    best_f1, epochs_no_improve = -1.0, 0
    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        for i in range(0, len(pool), ppo_cfg.batch_size):
            batch = pool[i : i + ppo_cfg.batch_size]
            prompts = [f"{x['text']}\nคำตอบ:" for x in batch]
            enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)

            gen = model.generate(**enc, max_new_tokens=120, do_sample=False)
            preds = tok.batch_decode(gen[:, enc.input_ids.shape[1]:],
                                     skip_special_tokens=True)

            rewards = torch.tensor([
                micro_f1(p, x["gold"]) * 10 for p, x in zip(preds, batch)
            ], device=model.device)

            trainer.step(prompts, preds, rewards)

        # ---- dev evaluation -------------------------------------------
        dev_preds = []
        for row in dev_small:
            enc_d = tok(f"{row['text']}\nคำตอบ:", return_tensors="pt").to(model.device)
            out = model.generate(**enc_d, max_new_tokens=120, do_sample=False)
            dev_preds.append(tok.decode(out[0, enc_d.input_ids.shape[1]:],
                                        skip_special_tokens=True))
        f1 = float(np.mean([micro_f1(p, g) for p, g in zip(dev_preds, dev_small["gold"])]))
        print(f"Epoch {epoch}: dev micro‑F1 = {f1:.3f}")

        # ---- early‑stopping logic ------------------------------------
        if f1 > best_f1 + 1e-4:
            best_f1 = f1
            epochs_no_improve = 0
            trainer.save_pretrained(out_path / "best")
            print("  ✔ new best, model saved →", out_path / "best")
        else:
            epochs_no_improve += 1
            if epochs_no_improve > args.patience:
                print("🔔 early stop (no improvement for", args.patience, "epochs)")
                break

        # also save every epoch (optional)
        trainer.save_pretrained(out_path / f"epoch{epoch}")

if __name__ == "__main__":
    main()
