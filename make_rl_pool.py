#!/usr/bin/env python
"""
Turn supervised file into rl_pool.jsonl for PPO/DPO.

Each output line:
{
  "text": "<prompt without trailing newline>",
  "gold": "<same JSON string you had in response_json>"
}
"""

import json, argparse, pathlib

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_file",  default="data/dataset/train_14k.jsonl",
                    help="original supervised jsonl")
    ap.add_argument("--out_file", default="data/dataset/rl_pool.jsonl")
    args = ap.parse_args()

    src = pathlib.Path(args.in_file)
    dst = pathlib.Path(args.out_file)
    dst.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with src.open(encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for raw in fin:
            obj = json.loads(raw)
            fout.write(json.dumps(
                {"text": obj["prompt"].rstrip(), "gold": obj["response_json"]},
                ensure_ascii=False) + "\n")
            written += 1

    print(f"✔  wrote {written:,} lines to {dst}")

if __name__ == "__main__":
    main()
