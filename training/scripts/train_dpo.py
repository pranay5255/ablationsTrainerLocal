#!/usr/bin/env python3
"""
Direct Preference Optimization (DPO) Training Script
====================================================
Uses TRL's DPOTrainer for preference-based fine-tuning.

Usage:
    python train_dpo.py --config experiments/ablations/smollm2_135m.yaml
    python train_dpo.py --model checkpoints/sft --dataset output/processed/dpo
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

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load experiment configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_model_for_dpo(
    model_path: str,
    max_seq_length: int = 2048,
    load_in_4bit: bool = False,
):
    """
    Setup model and tokenizer for DPO training.

    Args:
        model_path: Path to SFT checkpoint or base model
        max_seq_length: Maximum sequence length
        load_in_4bit: Use 4-bit quantization

    Returns:
        tuple: (model, tokenizer, ref_model)
    """
    try:
        from unsloth import FastLanguageModel
        use_unsloth = True
        logger.info("Using Unsloth for optimized DPO training")
    except ImportError:
        use_unsloth = False

    if use_unsloth:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
        )
        # DPO requires reference model - Unsloth handles this internally
        ref_model = None
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
        }

        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )

        # Check if it's a PEFT model
        if (Path(model_path) / "adapter_config.json").exists():
            # Load base model first
            from peft import AutoPeftModelForCausalLM
            model = AutoPeftModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

        # Reference model for DPO
        ref_model = None  # TRL can create this automatically

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, ref_model


def prepare_dpo_dataset(dataset_path: str) -> Dataset:
    """
    Load and prepare DPO dataset.

    Expected format:
    {
        "prompt": "Analyze the following...",
        "chosen": "Correct vulnerability analysis...",
        "rejected": "Incorrect analysis..."
    }
    """
    logger.info(f"Loading DPO dataset from {dataset_path}")

    if Path(dataset_path).exists():
        dataset = load_from_disk(dataset_path)
    else:
        from datasets import load_dataset
        dataset = load_dataset(dataset_path)

    if isinstance(dataset, dict) or hasattr(dataset, "keys"):
        train_dataset = dataset.get("train", dataset)
    else:
        train_dataset = dataset

    logger.info(f"Loaded {len(train_dataset)} preference pairs")
    return train_dataset


def train_dpo(
    model_path: str,
    dataset_path: str,
    output_dir: str,
    beta: float = 0.1,
    num_epochs: int = 1,
    learning_rate: float = 5e-7,
    per_device_batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    max_length: int = 2048,
    max_prompt_length: int = 1024,
    warmup_ratio: float = 0.1,
    save_steps: int = 200,
    eval_steps: int = 200,
    logging_steps: int = 50,
    load_in_4bit: bool = False,
    use_wandb: bool = True,
    wandb_project: str = "smart-contract-vuln-detection",
    wandb_run_name: Optional[str] = None,
    **kwargs
):
    """
    Run DPO training using TRL's DPOTrainer.

    Args:
        model_path: Path to SFT checkpoint
        dataset_path: Path to DPO dataset
        output_dir: Directory to save checkpoints
        beta: DPO beta parameter (controls deviation from reference)
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        per_device_batch_size: Batch size per GPU
        gradient_accumulation_steps: Gradient accumulation steps
        max_length: Maximum sequence length
        max_prompt_length: Maximum prompt length
        warmup_ratio: Warmup ratio
        save_steps: Save checkpoint every N steps
        eval_steps: Evaluate every N steps
        logging_steps: Log every N steps
        load_in_4bit: Use 4-bit quantization
        use_wandb: Enable W&B logging
        wandb_project: W&B project name
        wandb_run_name: W&B run name
    """
    from trl import DPOTrainer, DPOConfig

    # Setup model
    model, tokenizer, ref_model = setup_model_for_dpo(
        model_path=model_path,
        max_seq_length=max_length,
        load_in_4bit=load_in_4bit,
    )

    # Load dataset
    train_dataset = prepare_dpo_dataset(dataset_path)

    # Training arguments
    training_args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=warmup_ratio,
        save_steps=save_steps,
        eval_strategy="steps" if eval_steps > 0 else "no",
        eval_steps=eval_steps if eval_steps > 0 else None,
        logging_steps=logging_steps,
        beta=beta,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_8bit",
        save_total_limit=3,
        report_to="wandb" if use_wandb else "none",
        run_name=wandb_run_name,
        remove_unused_columns=False,
    )

    # Initialize trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    # Train
    logger.info("Starting DPO training...")
    logger.info(f"Beta: {beta}, LR: {learning_rate}")
    trainer.train()

    # Save final model
    logger.info(f"Saving DPO model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info("DPO training complete!")
    return trainer


def main():
    parser = argparse.ArgumentParser(description="DPO Training Script")

    parser.add_argument(
        "--config",
        type=str,
        help="Path to experiment config YAML"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="checkpoints/sft",
        help="Path to SFT checkpoint"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="output/processed/dpo",
        help="DPO dataset path"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/dpo",
        help="Output directory"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="DPO beta parameter"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of epochs"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-7,
        help="Learning rate"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
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

    args = parser.parse_args()

    if args.config:
        config = load_config(args.config)
        dpo_config = config.get("stages", {}).get("dpo", {})
        output_base = config.get("output", {}).get("dir", "checkpoints")

        # Get SFT model path
        sft_model_path = f"{output_base}/sft"
        if not Path(sft_model_path).exists():
            sft_model_path = config.get("model", {}).get("name", args.model)

        train_dpo(
            model_path=sft_model_path,
            dataset_path=dpo_config.get("dataset", args.dataset),
            output_dir=f"{output_base}/dpo",
            beta=dpo_config.get("beta", args.beta),
            num_epochs=dpo_config.get("num_epochs", args.epochs),
            learning_rate=dpo_config.get("learning_rate", args.lr),
            per_device_batch_size=dpo_config.get("per_device_batch_size", args.batch_size),
            gradient_accumulation_steps=dpo_config.get("gradient_accumulation_steps", args.grad_accum),
            max_length=dpo_config.get("max_length", 2048),
            max_prompt_length=dpo_config.get("max_prompt_length", 1024),
            warmup_ratio=dpo_config.get("warmup_ratio", 0.1),
            save_steps=dpo_config.get("save_steps", 200),
            eval_steps=dpo_config.get("eval_steps", 200),
            load_in_4bit=config.get("model", {}).get("load_in_4bit", args.load_in_4bit),
            use_wandb=not args.no_wandb,
            wandb_project=config.get("wandb", {}).get("project", "smart-contract-vuln-detection"),
            wandb_run_name=f"{config.get('experiment', {}).get('name', 'dpo')}-dpo",
        )
    else:
        train_dpo(
            model_path=args.model,
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            beta=args.beta,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            per_device_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            load_in_4bit=args.load_in_4bit,
            use_wandb=not args.no_wandb,
        )


if __name__ == "__main__":
    main()
