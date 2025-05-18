#!/usr/bin/env python3
import json

# Read the first line of the dataset
with open("data/dataset/teera_relation_extraction_mixed.jsonl", "r", encoding="utf-8") as f:
    first_line = f.readline().strip()
    
print("First line raw:")
print(first_line[:500])  # Print first 500 chars to avoid overwhelming output

try:
    data = json.loads(first_line)
    print("\nParsed JSON (first 5 keys):")
    for i, (key, value) in enumerate(data.items()):
        if i >= 5:
            break
        print(f"{key}: {str(value)[:100]}...")
except json.JSONDecodeError as e:
    print(f"\nError parsing JSON: {e}") 