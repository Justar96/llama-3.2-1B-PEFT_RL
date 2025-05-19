#!/usr/bin/env python
"""
Interactive chat interface for the ORE-1-lite model with developer-friendly features.

Features:
- Windows-safe implementation without readline dependency
- Configurable model parameters and generation settings
- JSON response parsing and formatting
- Streaming output with threading support
- Detailed error handling and logging

Usage:
    python chat_with_ore.py --model_dir <path_to_model> [--max_new_tokens 180] [--temperature 0.7] [--debug]
"""

import argparse
import json
import logging
import re
import sys
import torch
from threading import Thread
from typing import Tuple, Optional, Dict, Any

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextIteratorStreamer,
    PreTrainedModel,
    PreTrainedTokenizer
)

# Default configuration
DEFAULT_CONFIG = {
    "instruction": "Extract the relation from the following text:",
    "max_new_tokens": 180,
    "temperature": 0.7,
    "debug": False
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def load_model_and_tokenizer(model_dir: str) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
    """
    Load the model and tokenizer with 4-bit quantization.
    
    Args:
        model_dir: Path to the model directory
        
    Returns:
        Tuple of (tokenizer, model)
    """
    logger.info(f"Loading model from {model_dir}")
    
    # Configure 4-bit quantization
    qcfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    try:
        # Load tokenizer
        logger.debug("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        # Set padding token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.debug("Set pad_token to eos_token")
        
        # Load model with quantization
        logger.debug("Loading model with 4-bit quantization...")
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map="auto",
            quantization_config=qcfg,
            torch_dtype=torch.float16
        )
        model.eval()
        logger.info("Model and tokenizer loaded successfully")
        
        return tokenizer, model
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

def extract_json(text: str) -> str:
    """
    Extract and format the first valid JSON array from the text.
    
    Args:
        text: Input text potentially containing JSON
        
    Returns:
        Formatted JSON string if valid JSON found, otherwise original text
    """
    # Simple pattern to find JSON array - matches balanced brackets
    stack = []
    start_idx = -1
    
    for i, char in enumerate(text):
        if char == '[':
            if not stack:  # Found the start of a JSON array
                start_idx = i
            stack.append(char)
        elif char == ']' and stack:
            stack.pop()
            if not stack and start_idx != -1:  # Found the matching closing bracket
                json_str = text[start_idx:i+1]
                try:
                    # Validate and format JSON
                    json_obj = json.loads(json_str)
                    return json.dumps(json_obj, ensure_ascii=False, indent=2)
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON found: {e}")
                    return text.strip()
                except Exception as e:
                    logger.error(f"Error processing JSON: {e}")
                    return text.strip()
    
    logger.debug("No valid JSON array found in text")
    return text.strip()

def generate_response(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    text: str,
    config: Dict[str, Any]
) -> str:
    """
    Generate a response from the model for the given input text.
    
    Args:
        model: Loaded language model
        tokenizer: Tokenizer for the model
        text: Input text prompt
        config: Generation configuration
        
    Returns:
        Generated text response
    """
    logger.debug(f"Generating response for text: {text[:100]}...")
    
    # Format the prompt with instruction
    prompt = f"{config['instruction']}\n\"{text}\"\nคำตอบ:"
    
    # Tokenize input
    try:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    except Exception as e:
        logger.error(f"Tokenization failed: {e}")
        return "Error: Failed to process input"
    
    # Configure streaming
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_special_tokens=True,
        skip_prompt=True
    )
    
    # Start generation in a separate thread
    generation_kwargs = {
        **inputs,
        "max_new_tokens": config["max_new_tokens"],
        "temperature": config["temperature"],
        "do_sample": config["temperature"] > 0,
        "streamer": streamer,
    }
    
    thread = Thread(
        target=model.generate,
        kwargs=generation_kwargs,
        daemon=True
    )
    thread.start()
    
    # Stream the output
    generated_text = ""
    sys.stdout.write("🤖  Output> ")
    sys.stdout.flush()
    
    try:
        for chunk in streamer:
            sys.stdout.write(chunk)
            sys.stdout.flush()
            generated_text += chunk
    except Exception as e:
        logger.error(f"Error during generation: {e}")
        return f"Error during generation: {str(e)}"
    
    print("\n")
    return generated_text

def parse_arguments():
    """Parse and return command line arguments."""
    parser = argparse.ArgumentParser(
        description="Interactive chat with ORE-1-lite model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to the model directory"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=DEFAULT_CONFIG["max_new_tokens"],
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_CONFIG["temperature"],
        help="Sampling temperature (0.0 for greedy decoding)"
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default=DEFAULT_CONFIG["instruction"],
        help="Instruction prompt to use"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    return parser.parse_args()

def main():
    """Main entry point for the chat application."""
    # Parse arguments
    args = parse_arguments()
    
    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger.setLevel(log_level)
    
    # Prepare config
    config = {
        "instruction": args.instruction,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "debug": args.debug
    }
    
    logger.info(f"Starting chat with config: {config}")
    
    try:
        # Load model and tokenizer
        tokenizer, model = load_model_and_tokenizer(args.model_dir)
        
        # Display help
        print("\n" + "="*60)
        print("ORE-1-lite Chat Interface")
        print("Type your input and press Enter")
        print("Commands: 'exit', 'quit', or press Ctrl+C to exit")
        print("="*60 + "\n")
        
        # Main chat loop
        while True:
            try:
                # Get user input
                try:
                    user_input = input("📝  Input> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nExiting...")
                    break
                
                # Handle commands
                if not user_input:
                    continue
                if user_input.lower() in {"exit", "quit"}:
                    print("Goodbye!")
                    break
                
                # Generate response
                response = generate_response(model, tokenizer, user_input, config)
                
                # Display formatted JSON if present
                formatted_json = extract_json(response)
                if formatted_json != response.strip():
                    print("\nFormatted JSON output:")
                    print(formatted_json)
                
                print("\n" + "-" * 60)
                
            except Exception as e:
                logger.error(f"Error in chat loop: {e}")
                print(f"Error: {e}")
                continue
                
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
    
    logger.info("Chat session ended")

if __name__ == "__main__":
    main()
