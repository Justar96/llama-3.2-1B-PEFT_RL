#!/usr/bin/env python3
import json
from datasets import load_dataset

# Load the dataset
print("Loading dataset...")
try:
    ds = load_dataset("json", data_files="data/dataset/teera_relation_extraction_mixed.jsonl")["train"]
    print(f"Dataset loaded successfully with {len(ds)} examples")
    
    # Look at the first example
    first_example = ds[0]
    print("\nFirst example type:", type(first_example))
    print("First example attributes:", dir(first_example))
    
    # Get features/columns of the dataset
    print("\nDataset features:", ds.features)
    
    # Show structure of the first example
    print("\nFirst example keys:", list(first_example.keys()) if hasattr(first_example, 'keys') else "No keys (not a dict)")
    
    # If it's a dict, print the contents
    if hasattr(first_example, 'keys'):
        print("\nExample contents:")
        for key, value in first_example.items():
            print(f"{key}: {str(value)[:150]}...")
    else:
        print("\nRaw first example:", first_example)
        
    # Print the example as it would be used in the dataset
    print("\nAccessing first example attributes:")
    try:
        if hasattr(first_example, 'prompt'):
            print(f"prompt: {first_example.prompt[:150]}...")
        else:
            print("No 'prompt' attribute")
            
        if hasattr(first_example, 'response_json'):
            print(f"response_json: {first_example.response_json[:150]}...")
        else:
            print("No 'response_json' attribute")
    except Exception as e:
        print(f"Error accessing attributes: {e}")
        
except Exception as e:
    print(f"Error loading or processing dataset: {e}") 