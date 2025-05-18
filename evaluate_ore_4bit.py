#!/usr/bin/env python3
"""
4-bit Evaluation Script for ORE-1-lite

Loads a merged LoRA model in 4-bit, runs deterministic generation on a held-out split,
computes JSON-validity ratio, exact-match accuracy, micro precision/recall/F1,
average latency per prompt, and tokens/sec. Optionally evaluates the base model for comparison.
"""
import argparse
import time
import json
import warnings
import torch
from statistics import mean
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.metrics import precision_recall_fscore_support
import evaluate_ore1 as utils

def evaluate_model(model, tokenizer, eval_ds, device, max_new_tokens, label):
    json_valid, exact = 0, 0
    y_true, y_pred = [], []
    latencies, tokens_per_sec_list = [], []
    for row in tqdm(eval_ds, desc=f"Evaluating {label}"):
        prompt = row["prompt"] + "\nคำตอบ:"
        start = time.perf_counter()
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            ans_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
        latency = time.perf_counter() - start
        latencies.append(latency)
        ans_ids = ans_ids.cpu()
        tokens_per_sec_list.append(ans_ids.shape[-1] / latency)
        out_txt = tokenizer.decode(ans_ids[0], skip_special_tokens=True)
        pred = utils.extract_json(out_txt)
        ref = json.loads(row["response_json"])
        if isinstance(pred, list):
            json_valid += 1
            p_set = utils.triples_to_set(pred)
            r_set = utils.triples_to_set(ref)
            if p_set == r_set:
                exact += 1
            # Micro stats
            for r in r_set:
                found = any(utils.triples_match(p, r) for p in p_set)
                y_true.append(1); y_pred.append(1 if found else 0)
            for p in p_set:
                found = any(utils.triples_match(p, r) for r in r_set)
                if not found:
                    y_true.append(0); y_pred.append(1)
    n = len(eval_ds)
    json_ratio = json_valid / n
    exact_acc = exact / n
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    avg_latency = mean(latencies)
    avg_toks_per_sec = mean(tokens_per_sec_list)
    print(f"\n=== {label} Evaluation ===")
    print(f"Examples evaluated      : {n}")
    print(f"JSON-valid ratio        : {json_ratio:.2%}")
    print(f"Exact-match accuracy    : {exact_acc:.2%}")
    print(f"Micro Precision/Recall/F1: {p:.3f}/{r:.3f}/{f1:.3f}")
    print(f"Avg latency (sec)       : {avg_latency:.3f}")
    print(f"Tokens per sec          : {avg_toks_per_sec:.1f}")

def main():
    parser = argparse.ArgumentParser(description="4-bit Evaluation for ORE-1-lite")
    parser.add_argument("--model_dir", type=str, default="ORE-1-lite_merged", help="Path to merged LoRA model directory")
    parser.add_argument("--data_path", type=str, default="data/dataset/teera_relation_extraction_mixed.jsonl")
    parser.add_argument("--split_ratio", type=float, default=0.1, help="Held-out split ratio")
    parser.add_argument("--max_new_tokens", type=int, default=180, help="Max tokens to generate per prompt")
    parser.add_argument("--use_baseline", action="store_true", help="Also evaluate the base model for comparison")
    parser.add_argument("--base_model_name_or_path", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Base model identifier or path")
    args = parser.parse_args()

    # Load dataset and split
    ds = load_dataset("json", data_files=args.data_path)["train"]
    split = ds.train_test_split(test_size=args.split_ratio, seed=42)
    eval_ds = split["test"]

    # Prepare device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Silence generation warnings
    warnings.filterwarnings("ignore", message="`do_sample` is set to `False`.*")
    # Silence bitsandbytes Linear4bit compute dtype mismatch warning
    warnings.filterwarnings("ignore", message="Input type into Linear4bit is torch.float16, but bnb_4bit_compute_dtype=.*")
    # Silence pad_token_id auto-setting warnings
    warnings.filterwarnings("ignore", message="Setting `pad_token_id` to `eos_token_id`.*")

    # Load merged LoRA model in 4-bit
    print(f"Loading 4-bit merged model from {args.model_dir}...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16  # match input type for faster inference
    )
    lora_model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        quantization_config=quant_config,
        device_map="auto"
    )
    lora_model.config.pad_token_id = lora_model.config.eos_token_id
    lora_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    evaluate_model(lora_model, tokenizer, eval_ds, device, args.max_new_tokens, "LoRA-merged 4bit")

    # Optionally evaluate base model
    if args.use_baseline:
        print(f"\nLoading base model from {args.base_model_name_or_path}...")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model_name_or_path,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        base_model.config.pad_token_id = base_model.config.eos_token_id
        base_model.to(device).eval()
        tokenizer_base = AutoTokenizer.from_pretrained(args.base_model_name_or_path)
        evaluate_model(base_model, tokenizer_base, eval_ds, device, args.max_new_tokens, "Base model")

if __name__ == "__main__":
    main() 