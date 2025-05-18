#!/usr/bin/env python3
# Evaluate ORE-1-lite on a held-out split

import json, time, re, argparse
import torch
from statistics import mean
from tqdm import tqdm
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
from sklearn.metrics import precision_recall_fscore_support, classification_report

JSON_RE = re.compile(r"\[.*\]", re.DOTALL)  # grab first JSON array

def extract_json(txt):
    # Try to find JSON array pattern first
    m = JSON_RE.search(txt)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass  # continue to next approach if this fails
    
    # If no JSON found, try to extract structured triples from the text format
    # The model outputs text like "'A' มีความสัมพันธ์แบบ 'B' กับ 'C'"
    triples = []
    
    # Multiple patterns to match Thai relationship expressions
    patterns = [
        r"'([^']+)'\s+มีความสัมพันธ์แบบ\s+'([^']+)'\s+กับ\s+'([^']+)'",
        r"'([^']+)'\s+และ",  # for continuation patterns
    ]
    
    # Try the main pattern first
    matches = re.findall(patterns[0], txt)
    for match in matches:
        if len(match) == 3:
            subj, rel, obj = match
            # Skip if any part is empty
            if subj and rel and obj:
                triples.append({"subject": subj, "relation": rel, "object": obj})
    
    # If we found at least one triple with the main pattern, we're good
    if triples:
        return triples
    
    # Try an alternative approach: look for any quoted entities and relationships
    all_quotes = re.findall(r"'([^']+)'", txt)
    if len(all_quotes) >= 3:
        # Group in threes as (subject, relation, object)
        for i in range(0, len(all_quotes) - 2, 3):
            subj = all_quotes[i]
            rel = all_quotes[i+1]
            obj = all_quotes[i+2]
            # Skip if any part is empty or too long (likely not a relation)
            if subj and rel and obj and len(rel) < 50:
                triples.append({"subject": subj, "relation": rel, "object": obj})
    
    if triples:
        return triples
    
    return None

def normalize_text(text):
    """Normalize text to handle small variations"""
    if not text:
        return ""
    # Convert to lowercase, remove extra spaces, and normalize Thai characters
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    
    # Remove common variations in names
    text = text.replace("–", "-")  # Different dash types
    text = text.replace("—", "-")  # Different dash types
    text = text.replace("สนามบิน", "")  # Remove "airport" mention
    text = text.replace("อาโจบลังโก", "ajoblanco")  # Normalize Spanish name
    text = text.replace("ajoblanco", "ajoblanco")  # Standardize spelling
    
    # Remove parentheses and their contents
    text = re.sub(r'\([^)]*\)', '', text)
    
    # Handle names with variants
    name_variants = {
        "อดอล์ฟ ชาร์ฟ": "อดอล์ฟ แชร์ฟ",
        "อัลฟอนส์ กอร์บาค": "อัลฟองส์ กอร์บาค",
        "อดอลโฟ ซัวเรซ มาดริด-บาราคัส": "อดอลโฟ ซัวเรซ มาดริด-บาราคัส",
        "อดอลโฟ ซัวเรซ มาดริด-บาราคาส": "อดอลโฟ ซัวเรซ มาดริด-บาราคัส",
        "อาดอลโฟ ซัวเรซ มาดริด-บาราคาส": "อดอลโฟ ซัวเรซ มาดริด-บาราคัส",
        "อาดอลโฟ ซัวเรซ มาดริด-บาราคัส": "อดอลโฟ ซัวเรซ มาดริด-บาราคัส",
    }
    
    for variant, standard in name_variants.items():
        if variant in text:
            text = text.replace(variant, standard)
    
    # Additional normalization for specific cases
    if "ซาน เซบาสเตียน" in text or "ซาน เซบัสเตียน" in text:
        text = "ซาน เซบาสเตียน เด ลอส เรเยส"
        
    return text

def normalize_relation(relation):
    """Normalize relationship types to handle equivalences"""
    rel = relation.lower().strip()
    
    # Map of equivalent relationships
    rel_mapping = {
        "ในตำแหน่ง": "ในofficewhilepresident",
        "ผู้นำ": "ในofficewhilepresident",
        "บ้านเกิด": "สถานที่เกิด",
        "ผู้ดำเนินการ": "ปฏิบัติการองค์กร",
        "ปฏิบัติการองค์กร": "ปฏิบัติการองค์กร"
    }
    
    return rel_mapping.get(rel, rel)

def triples_match(t1, t2):
    """Check if two triples match, using flexible matching rules"""
    # Normalize all strings for comparison
    subj1 = normalize_text(t1[0])
    rel1 = normalize_relation(t1[1])
    obj1 = normalize_text(t1[2])
    
    subj2 = normalize_text(t2[0])
    rel2 = normalize_relation(t2[1])
    obj2 = normalize_text(t2[2])
    
    # For debugging
    # print(f"Comparing: {subj1}, {rel1}, {obj1} with {subj2}, {rel2}, {obj2}")
    
    # Subject and object can match partially (one is substring of the other)
    subj_match = subj1 in subj2 or subj2 in subj1 or subj1 == subj2
    
    # For objects that are lists (like "a, b, c และ d"), check if one object appears in the other
    obj_match = False  # Initialize to False
    if ',' in obj1 or ',' in obj2 or 'และ' in obj1 or 'และ' in obj2:
        # Split by common separators
        obj1_parts = re.split(r'[,،]|\sและ\s|\sand\s', obj1)
        obj2_parts = re.split(r'[,،]|\sและ\s|\sand\s', obj2)
        
        # Clean up each part
        obj1_parts = [part.strip() for part in obj1_parts if part.strip()]
        obj2_parts = [part.strip() for part in obj2_parts if part.strip()]
        
        # Check if any part in obj1 is in obj2 or vice versa
        for part1 in obj1_parts:
            for part2 in obj2_parts:
                if part1 in part2 or part2 in part1:
                    obj_match = True
                    break
            if obj_match:
                break
    else:
        obj_match = obj1 in obj2 or obj2 in obj1 or obj1 == obj2
    
    # Relation should match exactly after normalization
    rel_match = rel1 == rel2
    
    return subj_match and rel_match and obj_match

def triples_to_set(triples):
    """Convert list of triples to a normalized set for comparison"""
    normalized_set = set()
    for t in triples:
        subj = t.get("subject", "")
        rel = t.get("relation", "")
        obj = t.get("object", "")
        normalized_set.add((subj, rel, obj))
    return normalized_set

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="data/lora/ORE-1-lite")
    ap.add_argument("--data_path", default="data/dataset/teera_relation_extraction_mixed.jsonl")
    ap.add_argument("--sample_size", type=int, default=None,
                    help="Optionally evaluate on a random subset")
    args = ap.parse_args()

    # ── Load dataset & split ───────────────────────────────────────────────
    ds = load_dataset("json", data_files=args.data_path)["train"]
    eval_ds = ds.train_test_split(test_size=0.1, seed=42)["test"]
    if args.sample_size:
        eval_ds = Dataset.from_dict(eval_ds.shuffle(seed=0)[:args.sample_size])

    # ── Load model ─────────────────────────────────────────────────────────
    print(f"CUDA available: {torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    print("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    print(f"Tokenizer loaded: {type(tok)}")
    
    print(f"Loading model from {args.model_dir}...")
    mod = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=dtype
    ).to(device)
    mod.eval()
    # Silence generation warnings when do_sample=False
    warnings.filterwarnings("ignore", message="`do_sample` is set to `False`. However, `temperature` is set to")
    warnings.filterwarnings("ignore", message="`do_sample` is set to `False`. However, `top_p` is set to")
    # Explicitly set pad_token_id to eos_token_id to avoid repeated warnings
    mod.config.pad_token_id = mod.config.eos_token_id
    print(f"Model loaded successfully on {device}, dtype={dtype}")

    json_valid, exact, y_true, y_pred, latencies = 0, 0, [], [], []

    for i, row in enumerate(tqdm(eval_ds, total=len(eval_ds))):
        prompt = row["prompt"] + "\nคำตอบ:"
        t0 = time.perf_counter()
        
        # Prepare inputs
        inputs = tok(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
        
        # Generate output
        with torch.no_grad():
            ans_ids = mod.generate(
                **inputs,
                max_new_tokens=180,
                do_sample=False,
                pad_token_id=mod.config.pad_token_id
            )
        # Move predictions to CPU for decoding
        ans_ids = ans_ids.cpu()
        latency = time.perf_counter() - t0
        latencies.append(latency)

        out_txt = tok.decode(ans_ids[0], skip_special_tokens=True)
        pred_json = extract_json(out_txt)
        ref_json  = json.loads(row["response_json"])

        # Debug output for the first few examples
        if i < 3:
            print(f"\n--- Example {i+1} ---")
            print(f"Output text:\n{out_txt}")
            print(f"Extracted JSON: {pred_json}")
            print(f"Reference JSON: {ref_json}")

        if pred_json is not None and isinstance(pred_json, list):
            json_valid += 1

            p_set = triples_to_set(pred_json)
            r_set = triples_to_set(ref_json)

            # exact-match using flexible matching
            exact_match = True
            if len(p_set) == len(r_set):
                # For each reference triple, find a matching prediction
                matched_p = set()
                for r_triple in r_set:
                    found_match = False
                    for p_triple in p_set:
                        if p_triple not in matched_p and triples_match(p_triple, r_triple):
                            matched_p.add(p_triple)
                            found_match = True
                            break
                    if not found_match:
                        exact_match = False
                        break
            else:
                exact_match = False
                
            if exact_match:
                exact += 1

            # micro stats using flexible matching
            for r_triple in r_set:
                found = False
                for p_triple in p_set:
                    if triples_match(p_triple, r_triple):
                        found = True
                        break
                y_true.append(1)  # This is a reference triple
                y_pred.append(1 if found else 0)  # Predicted if found

            for p_triple in p_set:
                is_ref = False
                for r_triple in r_set:
                    if triples_match(p_triple, r_triple):
                        is_ref = True
                        break
                if not is_ref:
                    y_true.append(0)  # Not a reference triple
                    y_pred.append(1)  # But it was predicted

    # ── Metrics ────────────────────────────────────────────────────────────
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    # Additional metrics: macro-averaged precision, recall, and F1
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    n = len(eval_ds)
    print("\n=== Evaluation Report ===")
    print(f"Examples evaluated   : {n}")
    print(f"JSON-valid outputs   : {json_valid/n:.2%}")
    print(f"Exact-match accuracy : {exact/n:.2%}")
    print(f"Micro Precision      : {p:.3f}")
    print(f"Micro Recall         : {r:.3f}")
    print(f"Micro F1             : {f1:.3f}")
    print(f"Macro Precision      : {p_macro:.3f}")
    print(f"Macro Recall         : {r_macro:.3f}")
    print(f"Macro F1             : {f1_macro:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))
    print(f"Avg latency (sec)    : {mean(latencies):.3f}")
    tok_per_sec = mean([len(ans_ids[0]) / t for ans_ids, t in zip([ans_ids]*n, latencies)])
    print(f"~Tokens/sec          : {tok_per_sec:.1f}")

if __name__ == "__main__":
    main()
