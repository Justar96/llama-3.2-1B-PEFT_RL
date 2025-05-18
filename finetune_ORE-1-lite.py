# Fine-tuning LLaMA 3.2 1B on Teera Thai Relation Extraction
# Optimized Multi-GPU LoRA Fine-tuning

import os
import torch
import random
import argparse
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import load_dataset


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tokenize_fn(example, tokenizer, max_length: int):
    # Combine prompt + response fields
    prompt = example['prompt'] + "\nคำตอบ: " + example['response_text'] + tokenizer.eos_token
    tokenized = tokenizer(
        prompt,
        padding='max_length',
        truncation=True,
        max_length=max_length
    )
    # Labels same as input_ids
    tokenized['labels'] = tokenized['input_ids'].copy()
    return tokenized


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Get token predictions
    preds = np.argmax(logits, axis=-1)
    # Only consider non -100 labels
    mask = labels != -100
    labels_flat = labels[mask]
    preds_flat = preds[mask]
    # Compute accuracy
    accuracy = (preds_flat == labels_flat).astype(np.float32).mean().item()
    return {"accuracy": accuracy}


def main():
    parser = argparse.ArgumentParser(description="Optimized Fine-tune LLaMA-3.2-1B with LoRA")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--dataset_path", type=str, default="teera_relation_extraction_mixed.jsonl")
    parser.add_argument("--output_dir", type=str, default="ORE-1-optimized")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=12)  # Increased batch size
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)  # Reduced for higher throughput
    parser.add_argument("--lr", type=float, default=1e-4)  # Higher learning rate for LoRA
    parser.add_argument("--warmup_ratio", type=float, default=0.03)  # Shorter warmup
    parser.add_argument("--lora_r", type=int, default=32)  # Higher rank for better adaptation
    parser.add_argument("--lora_alpha", type=int, default=64)  # Higher alpha
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--use_bf16", action="store_true", default=True)
    parser.add_argument("--weight_decay", type=float, default=0.01)  # Add weight decay
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["linear", "cosine"])
    args = parser.parse_args()

    # Initialize distributed setup
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    print(f"Local Rank: {local_rank}, World Size: {world_size}")
    
    # Set the device based on local_rank
    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    set_seed(args.seed)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model and assign to specific device
    print("Loading model...")
    if local_rank == -1:
        device_map = "auto"
    else:
        device_map = {"": local_rank}
        
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if args.use_bf16 else torch.float16,
        device_map=device_map
    )
    
    print("Model loaded successfully")

    # Apply optimized LoRA config
    print("Applying LoRA with optimized settings...")
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()

    # Gradient checkpointing to save memory
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # Load dataset
    print("Loading dataset...")
    raw_dataset = load_dataset('json', data_files=args.dataset_path, split='train')
    # Split into train and validation
    split_datasets = raw_dataset.train_test_split(test_size=0.1, seed=args.seed)
    train_dataset = split_datasets['train']
    eval_dataset = split_datasets['test']

    # Tokenize dataset
    print("Tokenizing dataset...")
    train_dataset = train_dataset.map(
        lambda ex: tokenize_fn(ex, tokenizer, args.max_length),
        batched=False,
        remove_columns=train_dataset.column_names,
        num_proc=4  # Parallel processing
    )
    eval_dataset = eval_dataset.map(
        lambda ex: tokenize_fn(ex, tokenizer, args.max_length),
        batched=False,
        remove_columns=eval_dataset.column_names,
        num_proc=4
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Calculate training steps for scheduler
    steps_per_epoch = len(train_dataset) // (args.batch_size * world_size * args.gradient_accumulation_steps)
    total_training_steps = steps_per_epoch * args.epochs
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_strategy="epoch",
        learning_rate=args.lr,
        lr_scheduler_type=args.scheduler,
        warmup_ratio=args.warmup_ratio,
        logging_steps=10,
        save_steps=steps_per_epoch,  # Save once per epoch
        save_total_limit=2,  # Keep only the 2 most recent checkpoints
        fp16=not args.use_bf16,
        bf16=args.use_bf16,
        optim="adamw_torch",
        weight_decay=args.weight_decay,
        remove_unused_columns=False,
        torch_compile=False,
        report_to='none',
        # Advanced DDP settings
        local_rank=local_rank,
        ddp_find_unused_parameters=False,
        ddp_backend="nccl",
        # Performance optimizations
        dataloader_num_workers=4,  # Parallel data loading
        dataloader_drop_last=True,  # Drop incomplete batches
        gradient_checkpointing=False,  # Memory optimization
        # Half precision training
        half_precision_backend="auto",
    )

    print(f"Initializing Trainer with distributed setup: local_rank={local_rank}")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    # Launch training
    print(f"Starting training for {args.epochs} epochs, {total_training_steps} steps...")
    trainer.train()

    # Save model on the main process only
    if local_rank in [-1, 0]:
        print("Saving model...")
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save LoRA adapter only instead of full model (much smaller file)
        model.save_pretrained(args.output_dir)
        
        # Also save merged model if needed
        if os.environ.get("SAVE_MERGED_MODEL", "0") == "1":
            print("Creating and saving merged model...")
            merged_model = model.merge_and_unload()
            merged_model.save_pretrained(f"{args.output_dir}_merged")
            
        tokenizer.save_pretrained(args.output_dir)
        print(f"Training complete. Model saved to {args.output_dir}")


if __name__ == '__main__':
    # Run with: torchrun --nproc_per_node=NUM_GPUS finetune_ORE-1-optimized.py
    main()