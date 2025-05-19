#!/usr/bin/env python
"""
make_dev1k.py  –  sample a 1 000-line dev-set from a larger RL pool

Usage
-----
python make_dev1k.py --pool_file data/dataset/rl_pool.jsonl \
                     --output_file data/dataset/dev1k.jsonl \
                     --seed 42
"""
import argparse
import json
import random
from pathlib import Path

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pool_file", required=True,
                   help="Path to the full RL pool (JSONL with keys: text, gold)")
    p.add_argument("--output_file", default="dev1k.jsonl",
                   help="Where to write the sampled dev-set")
    p.add_argument("--n", type=int, default=1_000,
                   help="How many examples to sample")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for reproducibility")
    args = p.parse_args()

    random.seed(args.seed)

    # read the full pool
    with open(args.pool_file, "r", encoding="utf-8") as f:
        pool = [json.loads(line) for line in f]

    if len(pool) < args.n:
        raise ValueError(f"Pool has only {len(pool)} lines – can't sample {args.n}")

    sample = random.sample(pool, k=args.n)

    # write dev1k
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        for row in sample:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"✅ Wrote {args.n} examples to {args.output_file}")

if __name__ == "__main__":
    main()
