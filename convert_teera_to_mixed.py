#!/usr/bin/env python3
# Convert Teera/RelationExtraction-NLG-Thai to mixed-format JSONL
#  – safe comma-splitting
#  – backslash clean-up
#  – prompt variety (Thai & English)

import json
import re
import random
from datasets import load_dataset


# ------------------------------------------------------------------
# Prompt templates (add / edit as you wish)
# ------------------------------------------------------------------
PROMPT_TEMPLATES = [
    "ถอดความสัมพันธ์จากข้อความต่อไปนี้:",
    "โปรดระบุความสัมพันธ์จากข้อความนี้:",
    "ข้อความนี้แสดงถึงความสัมพันธ์อะไร?",
    "Identify the relationship expressed in this text:",
    "Extract the relation from the following text:"
]

# ------------------------------------------------------------------
# Regex to split on commas that separate each triplet
#   –  ,␣ followed by something that still contains ' --> '
#   – avoids splitting if the comma lives inside a field
# ------------------------------------------------------------------
COMMA_SPLIT_RE = re.compile(r', (?=[^][]*?--> )')


def split_triplets(raw: str):
    """Return a list of raw 'A --> B --> C' strings."""
    return COMMA_SPLIT_RE.split(raw.strip()[1:-1])  # remove outer [ ] first


def parse_triplet(tp: str):
    """Return (subj, rel, obj) or None if malformed."""
    parts = tp.split(" --> ")
    if len(parts) != 3:
        return None
    subj, rel, obj = (p.strip() for p in parts)
    # Clean escaped back-slashes
    obj = obj.lstrip("\\").replace("\\\\", "\\")
    return subj, rel, obj


def main():
    ds = load_dataset("Teera/RelationExtraction-NLG-Thai", split="train")

    out_path = "teera_relation_extraction_mixed_improved.jsonl"
    written = 0

    with open(out_path, "w", encoding="utf-8") as fout:
        for item in ds:
            text = item.get("text", "").strip()
            triple_field = item.get("triple", "").strip()
            if not text or not triple_field:
                continue

            json_rels, text_rels = [], []
            for raw_trip in split_triplets(triple_field):
                parsed = parse_triplet(raw_trip)
                if not parsed:
                    continue
                s, r, o = parsed
                json_rels.append({"subject": s, "relation": r, "object": o})
                text_rels.append(f"'{s}' มีความสัมพันธ์แบบ '{r}' กับ '{o}'")

            if not json_rels:
                continue

            prompt = f"{random.choice(PROMPT_TEMPLATES)}\n\"{text}\""
            record = {
                "prompt": prompt,
                "response_json": json.dumps(json_rels, ensure_ascii=False),
                "response_text": " และ ".join(text_rels)
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"✅ Finished. Wrote {written} examples → {out_path}")


if __name__ == "__main__":
    main()
