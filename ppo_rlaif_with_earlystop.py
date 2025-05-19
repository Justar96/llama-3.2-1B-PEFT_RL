#!/usr/bin/env python
"""
Metric-based PPO (RLAIF) for ORE-1-lite **with automatic early-stopping**

Compatible with **trl 0.11.x** (Option B) – now with:

* tqdm progress bars
* automatic 1 000-example dev-set if --dev_file is omitted
* warning filters to hide TRL v2 deprecation spam
* efficient batch encoding (fits on an 8–12 GB GPU)

Run:
python ppo_rlaif_with_earlystop.py \
       --model_dir  ORE-1-lite_merged \
       --pool_file  data/dataset/rl_pool.jsonl \
       --dev_file   data/dataset/dev1k.jsonl
"""
"""
Metric-based PPO for ORE-1-lite (TRL 0.11.x) – single-GPU variant
"""
import argparse
import os
import random
import warnings
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.quantization_config import BitsAndBytesConfig
from trl import AutoModelForCausalLMWithValueHead
from trl.trainer import PPOConfig, PPOTrainer

warnings.filterwarnings("ignore", category=FutureWarning, module="trl")
warnings.filterwarnings(
    "ignore", category=UserWarning, message="No dataset is provided.*"
)


# ----------------- utility --------------------------------------------------
def micro_f1(pred, ref):
    p, r = pred.split(), ref.split()
    o = set(p) & set(r)
    if not o:
        return 0.0
    prec, rec = len(o) / len(p), len(o) / len(r)
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


# --------------------------------------------------------------------------- #
# main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> None:
    # ----------------- CLI ------------------------------------------------------
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--pool_file", required=True)
    ap.add_argument("--dev_file")
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--output_dir", default="ppo_rlaif_runs")
    ap.add_argument("--gpu", type=int, default=0, help="CUDA device index (default 0)")
    args = ap.parse_args()

    # ----------------- choose GPU ----------------------------------------------
    # For multi-GPU: use Accelerate or torchrun, and let device_map be handled automatically.
    # Remove manual device selection if using Accelerate/torchrun.
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available –- cannot select GPU")
    if args.gpu < 0 or args.gpu >= torch.cuda.device_count():
        raise RuntimeError(
            f"Invalid GPU index {args.gpu}. Available CUDA devices: 0 to {torch.cuda.device_count()-1}"
        )
    torch.cuda.set_device(args.gpu)  # make this the default
    device_name = torch.cuda.get_device_name(args.gpu)
    print(f"✓ using cuda:{args.gpu}  ({device_name})")
    # If using multi-GPU, comment out the above block and run with Accelerate or torchrun.

    # ----------------- tokenizer & model ---------------------------------------
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    tok.padding_side, tok.pad_token = "left", tok.eos_token

    qcfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        device_map="auto",  # <-- Use 'auto' for multi-GPU/model parallelism
        quantization_config=qcfg,
    )
    model = AutoModelForCausalLMWithValueHead.from_pretrained(base)
    device = next(model.parameters()).device
    # For multi-GPU, do not assert device here. Let Accelerate/DDP handle device placement.

    # ---- PPO config ---------------------------------------------------------
    ppo_cfg = PPOConfig(
        learning_rate=2e-6,
        batch_size=16,
        mini_batch_size=4,
        gradient_accumulation_steps=4,
    )
    trainer = PPOTrainer(
        config=ppo_cfg,
        model=model,
        ref_model=None,
        tokenizer=tok,
    )

    # ---- datasets -----------------------------------------------------------
    pool_ds = list(load_dataset("json", data_files=args.pool_file, split="train"))

    dev_ds = None
    if args.dev_file:
        dev_ds = list(load_dataset("json", data_files=args.dev_file, split="train"))

    # ---- training loop ------------------------------------------------------
    best_f1 = -1.0
    epochs_no_improv = 0
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(1, 100 + 1):
        print(f"\n========== Epoch {epoch} ==========")

        for b_start in tqdm(
            range(0, len(pool_ds), ppo_cfg.batch_size),
            desc="train-batches",
            unit_scale=ppo_cfg.batch_size,
        ):
            batch = pool_ds[b_start : b_start + ppo_cfg.batch_size]
            prompts = [row["text"] for row in batch]

            # encode once
            input_batch = tok(prompts, return_tensors="pt", padding=True).to(device)
            query_tensors = list(input_batch.input_ids)

            # deterministic generation
            with torch.no_grad():
                output = model.generate(
                    **input_batch,
                    max_new_tokens=120,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )

            # strip prompts & decode
            decoded = []
            for prompt_ids, full in zip(input_batch.input_ids, output):
                resp_ids = full[len(prompt_ids) :]
                decoded.append(tok.decode(resp_ids, skip_special_tokens=True))

            # rewards
            rewards = [
                torch.tensor(micro_f1(p, g["gold"]), device=device)
                for p, g in zip(decoded, batch)
            ]

            trainer.step(query_tensors, list(output), rewards)

        # ---- dev eval --------------------------------------------------------
        dev_sample = (
            dev_ds
            if dev_ds is not None
            else random.sample(pool_ds, k=min(1_000, len(pool_ds)))
        )

        dev_preds = []
        for row in tqdm(dev_sample, desc="dev-gen"):
            prompt = f'{row["text"]}\nคำตอบ:'
            with torch.no_grad():
                out = model.generate(
                    **tok(prompt, return_tensors="pt").to(device),
                    max_new_tokens=120,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )
            gen = tok.decode(out[0], skip_special_tokens=True)[len(prompt) :]
            dev_preds.append(gen)

        avg_f1 = sum(
            micro_f1(p, r["gold"]) for p, r in zip(dev_preds, dev_sample)
        ) / len(dev_sample)
        print(f"[dev] micro-F1 = {avg_f1:.4f}")

        # ---- early-stop ------------------------------------------------------
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            epochs_no_improv = 0
            ckpt_dir = Path(args.output_dir) / "best"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            print(f"🔥  new best → saved to {ckpt_dir}")
        else:
            epochs_no_improv += 1
            if epochs_no_improv >= args.patience:
                print(f"⏹  no improvement for {args.patience} epoch(s) – stopping.")
                break


if __name__ == "__main__":
    main()
