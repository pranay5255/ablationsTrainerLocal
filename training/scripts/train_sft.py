#!/usr/bin/env python3
"""
Supervised Fine-Tuning (SFT) Training Script
=============================================
Uses Unsloth + TRL for 2x speed and 60% less memory on RTX 4090.

Usage:
    python train_sft.py --config experiments/ablations/smollm2_135m.yaml
    python train_sft.py --model HuggingFaceTB/SmolLM2-135M --dataset output/processed/sft
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import yaml
import torch
from datasets import load_from_disk, Dataset

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load experiment configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_model_and_tokenizer(
    model_name: str,
    max_seq_length: int = 2048,
    dtype: str = "bfloat16",
    load_in_4bit: bool = False,
    lora_config: Optional[Dict] = None,
):
    """
    Setup model and tokenizer using Unsloth for optimization.

    Returns:
        tuple: (model, tokenizer)
    """
    try:
        from unsloth import FastLanguageModel
        use_unsloth = True
        logger.info("Using Unsloth for optimized training")
    except ImportError:
        logger.warning("Unsloth not available, falling back to standard transformers")
        use_unsloth = False

    if use_unsloth:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=getattr(torch, dtype) if isinstance(dtype, str) else dtype,
            load_in_4bit=load_in_4bit,
        )

        # Apply LoRA if configured
        if lora_config and lora_config.get("enabled", True):
            model = FastLanguageModel.get_peft_model(
                model,
                r=lora_config.get("r", 16),
                lora_alpha=lora_config.get("alpha", 32),
                lora_dropout=lora_config.get("dropout", 0.05),
                target_modules=lora_config.get("target_modules", [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"
                ]),
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=42,
            )
    else:
        # Fallback to standard transformers + PEFT
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": getattr(torch, dtype) if isinstance(dtype, str) else dtype,
        }

        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        if lora_config and lora_config.get("enabled", True):
            peft_config = LoraConfig(
                r=lora_config.get("r", 16),
                lora_alpha=lora_config.get("alpha", 32),
                lora_dropout=lora_config.get("dropout", 0.05),
                target_modules=lora_config.get("target_modules"),
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, peft_config)

    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def prepare_dataset(
    dataset_path: str,
    tokenizer,
    max_seq_length: int = 2048,
) -> Dataset:
    """
    Load and prepare dataset for SFT training.

    Expects dataset with 'messages' field in chat format.
    """
    logger.info(f"Loading dataset from {dataset_path}")

    if Path(dataset_path).exists():
        dataset = load_from_disk(dataset_path)
    else:
        from datasets import load_dataset
        dataset = load_dataset(dataset_path)

    # Get train split
    if isinstance(dataset, dict) or hasattr(dataset, "keys"):
        if "train" in dataset:
            train_dataset = dataset["train"]
        else:
            train_dataset = dataset
    else:
        train_dataset = dataset

    logger.info(f"Loaded {len(train_dataset)} training examples")
    return train_dataset


def train_sft(
    model_name: str,
    dataset_path: str,
    output_dir: str,
    num_epochs: int = 2,
    learning_rate: float = 2e-5,
    per_device_batch_size: int = 4,
    gradient_accumulation_steps: int = 8,
    max_seq_length: int = 2048,
    warmup_ratio: float = 0.0,
    weight_decay: float = 0.01,
    save_steps: int = 500,
    eval_steps: int = 500,
    logging_steps: int = 50,
    load_in_4bit: bool = False,
    lora_config: Optional[Dict] = None,
    use_wandb: bool = True,
    wandb_project: str = "smart-contract-vuln-detection",
    wandb_run_name: Optional[str] = None,
    **kwargs
):
    """
    Run SFT training using TRL's SFTTrainer.

    Args:
        model_name: HuggingFace model identifier
        dataset_path: Path to training dataset
        output_dir: Directory to save checkpoints
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        per_device_batch_size: Batch size per GPU
        gradient_accumulation_steps: Gradient accumulation steps
        max_seq_length: Maximum sequence length
        warmup_ratio: Warmup ratio
        weight_decay: Weight decay
        save_steps: Save checkpoint every N steps
        eval_steps: Evaluate every N steps
        logging_steps: Log every N steps
        load_in_4bit: Use 4-bit quantization
        lora_config: LoRA configuration dict
        use_wandb: Enable Weights & Biases logging
        wandb_project: W&B project name
        wandb_run_name: W&B run name
    """
    from trl import SFTTrainer, SFTConfig

    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        lora_config=lora_config,
    )

    # Load dataset
    train_dataset = prepare_dataset(dataset_path, tokenizer, max_seq_length)

    # Setup training arguments
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        save_steps=save_steps,
        eval_strategy="steps" if eval_steps > 0 else "no",
        eval_steps=eval_steps if eval_steps > 0 else None,
        logging_steps=logging_steps,
        max_seq_length=max_seq_length,
        packing=False,  # Disable packing for vulnerability detection
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_8bit",
        save_total_limit=3,
        report_to="wandb" if use_wandb else "none",
        run_name=wandb_run_name,
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    # Train
    logger.info("Starting SFT training...")
    trainer.train()

    # Save final model
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info("SFT training complete!")
    return trainer


def main():
    parser = argparse.ArgumentParser(description="SFT Training Script")

    parser.add_argument(
        "--config",
        type=str,
        help="Path to experiment config YAML"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="HuggingFaceTB/SmolLM2-135M",
        help="Model name or path"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="output/processed/sft",
        help="Dataset path"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/sft",
        help="Output directory"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of epochs"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per-device batch size"
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=8,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Use 4-bit quantization"
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="smart-contract-vuln-detection",
        help="W&B project name"
    )

    args = parser.parse_args()

    # Load config if provided
    if args.config:
        config = load_config(args.config)
        model_name = config.get("model", {}).get("name", args.model)
        sft_config = config.get("stages", {}).get("sft", {})
        lora_config = config.get("lora", {})
        output_dir = config.get("output", {}).get("dir", args.output_dir)
        wandb_config = config.get("wandb", {})

        train_sft(
            model_name=model_name,
            dataset_path=sft_config.get("dataset", args.dataset),
            output_dir=f"{output_dir}/sft",
            num_epochs=sft_config.get("num_epochs", args.epochs),
            learning_rate=sft_config.get("learning_rate", args.lr),
            per_device_batch_size=sft_config.get("per_device_batch_size", args.batch_size),
            gradient_accumulation_steps=sft_config.get("gradient_accumulation_steps", args.grad_accum),
            max_seq_length=config.get("model", {}).get("max_seq_length", 2048),
            warmup_ratio=sft_config.get("warmup_ratio", 0.0),
            weight_decay=sft_config.get("weight_decay", 0.01),
            save_steps=sft_config.get("save_steps", 500),
            eval_steps=sft_config.get("eval_steps", 500),
            logging_steps=sft_config.get("logging_steps", 50),
            load_in_4bit=config.get("model", {}).get("load_in_4bit", args.load_in_4bit),
            lora_config=lora_config,
            use_wandb=not args.no_wandb,
            wandb_project=wandb_config.get("project", args.wandb_project),
            wandb_run_name=config.get("experiment", {}).get("name"),
        )
    else:
        # Use command line args
        train_sft(
            model_name=args.model,
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            per_device_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            load_in_4bit=args.load_in_4bit,
            use_wandb=not args.no_wandb,
            wandb_project=args.wandb_project,
        )


if __name__ == "__main__":
    main()
