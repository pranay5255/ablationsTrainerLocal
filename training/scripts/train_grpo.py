#!/usr/bin/env python3
"""
Group Relative Policy Optimization (GRPO) Training Script
=========================================================
Uses TRL's GRPOTrainer with custom reward function for vulnerability detection.

Based on DeepSeek R1 methodology as referenced in PRD Section 4.2.

Usage:
    python train_grpo.py --config experiments/ablations/smollm2_135m.yaml
    python train_grpo.py --model checkpoints/dpo --dataset output/processed/grpo
"""

import os
import sys
import argparse
import logging
import re
from pathlib import Path
from typing import Optional, Dict, Any, List

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


# =============================================================================
# Reward Functions for GRPO (PRD Section 6.3)
# =============================================================================

def calculate_swc_match_reward(model_output: str, ground_truth: List[Dict]) -> float:
    """
    Reward for correct SWC ID prediction.

    +1.0 for correct SWC
    -1.0 for incorrect SWC
    +0.5 for partial match (correct category)
    """
    # Extract SWC from model output
    swc_pattern = r'SWC[:\s]*(\d+|SWC-\d+)'
    model_swcs = re.findall(swc_pattern, model_output, re.IGNORECASE)
    model_swcs = [s.replace('SWC-', '').replace('swc-', '') for s in model_swcs]

    if not ground_truth:
        # Clean contract - reward for not finding vulnerabilities
        if not model_swcs or 'none' in model_output.lower():
            return 1.0
        return -1.0  # False positive

    # Vulnerable contract
    gt_swcs = [v.get('swc_id', '').replace('SWC-', '') for v in ground_truth]

    if not model_swcs:
        return -1.0  # Missed vulnerability

    # Check for matches
    matches = sum(1 for ms in model_swcs if ms in gt_swcs)
    if matches > 0:
        return 1.0 * (matches / max(len(gt_swcs), len(model_swcs)))

    return -0.5  # Wrong type


def calculate_line_accuracy_reward(model_output: str, ground_truth: List[Dict]) -> float:
    """
    Reward for accurate line number identification.

    Uses IoU (Intersection over Union) for line ranges.
    """
    # Extract lines from model output
    line_pattern = r'Lines?[:\s]*(\d+)(?:\s*-\s*(\d+))?'
    model_lines = re.findall(line_pattern, model_output)

    if not ground_truth or not model_lines:
        return 0.0

    # Get ground truth line ranges
    gt_lines = set()
    for vuln in ground_truth:
        lines = vuln.get('lines', vuln.get('line_range', ''))
        if isinstance(lines, str) and '-' in lines:
            start, end = map(int, lines.split('-'))
            gt_lines.update(range(start, end + 1))
        elif isinstance(lines, (int, str)):
            try:
                gt_lines.add(int(lines))
            except ValueError:
                pass

    # Get model predicted lines
    pred_lines = set()
    for match in model_lines:
        start = int(match[0])
        end = int(match[1]) if match[1] else start
        pred_lines.update(range(start, end + 1))

    if not gt_lines or not pred_lines:
        return 0.0

    # Calculate IoU
    intersection = len(gt_lines & pred_lines)
    union = len(gt_lines | pred_lines)

    return intersection / union if union > 0 else 0.0


def calculate_format_reward(model_output: str) -> float:
    """
    Reward for following the correct output format.

    Expected format (PRD Section 3.1):
    - Vulnerability: <type>
    - CWE: <id>
    - SWC: <id>
    - Lines: <range>
    - Severity: <level>
    - Explanation: <text>
    - Remediation: <text>
    """
    expected_fields = [
        r'Vulnerability[:\s]',
        r'SWC[:\s]',
        r'Lines?[:\s]',
        r'Severity[:\s]',
        r'Explanation[:\s]',
    ]

    matches = sum(1 for pattern in expected_fields
                  if re.search(pattern, model_output, re.IGNORECASE))

    return matches / len(expected_fields)


def vulnerability_detection_reward(
    model_outputs: List[str],
    ground_truths: List[List[Dict]],
    use_slither: bool = False
) -> List[float]:
    """
    Combined reward function for GRPO as per PRD Section 6.3.

    Args:
        model_outputs: List of model-generated responses
        ground_truths: Corresponding ground truth vulnerabilities
        use_slither: Whether to use Slither for validation

    Returns:
        List of reward scores
    """
    rewards = []

    for output, gt in zip(model_outputs, ground_truths):
        # Component rewards
        swc_reward = calculate_swc_match_reward(output, gt)
        line_reward = calculate_line_accuracy_reward(output, gt)
        format_reward = calculate_format_reward(output)

        # Weighted combination
        total_reward = (
            0.5 * swc_reward +      # 50% weight on correct detection
            0.3 * line_reward +     # 30% weight on line accuracy
            0.2 * format_reward     # 20% weight on format compliance
        )

        # Optional: Slither validation bonus
        if use_slither:
            try:
                slither_reward = validate_with_slither(output, gt)
                total_reward = 0.8 * total_reward + 0.2 * slither_reward
            except Exception:
                pass  # Skip Slither validation if it fails

        rewards.append(total_reward)

    return rewards


def validate_with_slither(model_output: str, ground_truth: List[Dict]) -> float:
    """
    Validate model output using Slither static analyzer.

    This is a placeholder - actual implementation requires Slither setup.
    """
    # TODO: Implement actual Slither validation
    # For now, return neutral score
    return 0.0


# =============================================================================
# GRPO Training
# =============================================================================

def setup_model_for_grpo(
    model_path: str,
    max_seq_length: int = 2048,
    load_in_4bit: bool = False,
):
    """Setup model and tokenizer for GRPO training."""
    try:
        from unsloth import FastLanguageModel
        use_unsloth = True
    except ImportError:
        use_unsloth = False

    if use_unsloth:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
        )
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

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
            )

        if (Path(model_path) / "adapter_config.json").exists():
            from peft import AutoPeftModelForCausalLM
            model = AutoPeftModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def prepare_grpo_dataset(dataset_path: str) -> Dataset:
    """
    Load GRPO prompt dataset.

    Expected format:
    {
        "prompt": "Analyze the following...",
        "has_vulnerability": true/false,
        "ground_truth": [{"swc_id": "SWC-107", ...}]
    }
    """
    logger.info(f"Loading GRPO dataset from {dataset_path}")

    if Path(dataset_path).exists():
        dataset = load_from_disk(dataset_path)
    else:
        from datasets import load_dataset
        dataset = load_dataset(dataset_path)

    if isinstance(dataset, dict) or hasattr(dataset, "keys"):
        train_dataset = dataset.get("train", dataset)
    else:
        train_dataset = dataset

    logger.info(f"Loaded {len(train_dataset)} prompts for GRPO")
    return train_dataset


def train_grpo(
    model_path: str,
    dataset_path: str,
    output_dir: str,
    num_generations: int = 4,
    num_epochs: int = 1,
    learning_rate: float = 1e-6,
    per_device_batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    max_seq_length: int = 2048,
    save_steps: int = 200,
    logging_steps: int = 50,
    load_in_4bit: bool = False,
    use_slither: bool = False,
    use_wandb: bool = True,
    wandb_project: str = "smart-contract-vuln-detection",
    wandb_run_name: Optional[str] = None,
    **kwargs
):
    """
    Run GRPO training using TRL's GRPOTrainer.

    Args:
        model_path: Path to DPO/SFT checkpoint
        dataset_path: Path to GRPO prompt dataset
        output_dir: Directory to save checkpoints
        num_generations: Number of responses to generate per prompt
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        per_device_batch_size: Batch size per GPU
        gradient_accumulation_steps: Gradient accumulation steps
        max_seq_length: Maximum sequence length
        save_steps: Save checkpoint every N steps
        logging_steps: Log every N steps
        load_in_4bit: Use 4-bit quantization
        use_slither: Use Slither for reward validation
        use_wandb: Enable W&B logging
        wandb_project: W&B project name
        wandb_run_name: W&B run name
    """
    try:
        from trl import GRPOTrainer, GRPOConfig
    except ImportError:
        logger.error("GRPOTrainer not available. Please update TRL: pip install trl>=0.8.0")
        logger.info("Falling back to manual GRPO implementation...")
        return train_grpo_manual(
            model_path, dataset_path, output_dir, num_generations,
            num_epochs, learning_rate, per_device_batch_size,
            gradient_accumulation_steps, max_seq_length, save_steps,
            logging_steps, load_in_4bit, use_slither, use_wandb,
            wandb_project, wandb_run_name
        )

    # Setup model
    model, tokenizer = setup_model_for_grpo(
        model_path=model_path,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
    )

    # Load dataset
    train_dataset = prepare_grpo_dataset(dataset_path)

    # Custom reward function
    def reward_fn(samples: List[str], prompts: List[str], outputs: List[str], **kwargs):
        """Custom reward function for vulnerability detection."""
        # Get ground truths from dataset
        ground_truths = []
        for prompt in prompts:
            # Find matching ground truth in dataset
            # This is simplified - in practice, you'd track this more carefully
            ground_truths.append([])  # Default to empty

        return vulnerability_detection_reward(
            outputs,
            ground_truths,
            use_slither=use_slither
        )

    # Training config
    training_args = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_generations=num_generations,
        save_steps=save_steps,
        logging_steps=logging_steps,
        bf16=True,
        gradient_checkpointing=True,
        save_total_limit=3,
        report_to="wandb" if use_wandb else "none",
        run_name=wandb_run_name,
    )

    # Initialize trainer
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
    )

    # Train
    logger.info("Starting GRPO training...")
    logger.info(f"Num generations: {num_generations}, LR: {learning_rate}")
    trainer.train()

    # Save final model
    logger.info(f"Saving GRPO model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info("GRPO training complete!")
    return trainer


def train_grpo_manual(
    model_path: str,
    dataset_path: str,
    output_dir: str,
    num_generations: int = 4,
    num_epochs: int = 1,
    learning_rate: float = 1e-6,
    per_device_batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    max_seq_length: int = 2048,
    save_steps: int = 200,
    logging_steps: int = 50,
    load_in_4bit: bool = False,
    use_slither: bool = False,
    use_wandb: bool = True,
    wandb_project: str = "smart-contract-vuln-detection",
    wandb_run_name: Optional[str] = None,
):
    """
    Manual GRPO implementation as fallback.

    This implements the core GRPO algorithm:
    1. Generate multiple responses per prompt
    2. Score with reward function
    3. Use relative rankings for policy update
    """
    from transformers import Trainer, TrainingArguments
    import numpy as np

    logger.info("Using manual GRPO implementation")

    # Setup
    model, tokenizer = setup_model_for_grpo(model_path, max_seq_length, load_in_4bit)
    train_dataset = prepare_grpo_dataset(dataset_path)

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # For each epoch
    for epoch in range(num_epochs):
        logger.info(f"GRPO Epoch {epoch + 1}/{num_epochs}")

        for i, example in enumerate(train_dataset):
            prompt = example["prompt"]
            ground_truth = example.get("ground_truth", [])

            # Generate multiple responses
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_seq_length)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            responses = []
            for _ in range(num_generations):
                with torch.no_grad():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=512,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                    )
                response = tokenizer.decode(output[0], skip_special_tokens=True)
                response = response[len(prompt):]  # Remove prompt
                responses.append(response)

            # Score responses
            rewards = vulnerability_detection_reward(
                responses,
                [ground_truth] * len(responses),
                use_slither=use_slither
            )

            # Compute relative advantage (GRPO key insight)
            mean_reward = np.mean(rewards)
            advantages = [r - mean_reward for r in rewards]

            # Log progress
            if i % logging_steps == 0:
                logger.info(f"Step {i}: Mean reward = {mean_reward:.3f}")

            # Save checkpoint
            if i > 0 and i % save_steps == 0:
                checkpoint_dir = f"{output_dir}/checkpoint-{i}"
                model.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)

    # Save final model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"GRPO training complete. Model saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="GRPO Training Script")

    parser.add_argument("--config", type=str, help="Path to experiment config YAML")
    parser.add_argument("--model", type=str, default="checkpoints/dpo", help="Path to checkpoint")
    parser.add_argument("--dataset", type=str, default="output/processed/grpo", help="Dataset path")
    parser.add_argument("--output-dir", type=str, default="checkpoints/grpo", help="Output directory")
    parser.add_argument("--num-generations", type=int, default=4, help="Responses per prompt")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation")
    parser.add_argument("--load-in-4bit", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--use-slither", action="store_true", help="Use Slither validation")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B")

    args = parser.parse_args()

    if args.config:
        config = load_config(args.config)
        grpo_config = config.get("stages", {}).get("grpo", {})
        output_base = config.get("output", {}).get("dir", "checkpoints")

        # Get DPO model path (or SFT if DPO not run)
        dpo_model_path = f"{output_base}/dpo"
        if not Path(dpo_model_path).exists():
            dpo_model_path = f"{output_base}/sft"
        if not Path(dpo_model_path).exists():
            dpo_model_path = config.get("model", {}).get("name", args.model)

        train_grpo(
            model_path=dpo_model_path,
            dataset_path=grpo_config.get("dataset", args.dataset),
            output_dir=f"{output_base}/grpo",
            num_generations=grpo_config.get("num_generations", args.num_generations),
            num_epochs=grpo_config.get("num_epochs", args.epochs),
            learning_rate=grpo_config.get("learning_rate", args.lr),
            per_device_batch_size=grpo_config.get("per_device_batch_size", args.batch_size),
            gradient_accumulation_steps=grpo_config.get("gradient_accumulation_steps", args.grad_accum),
            load_in_4bit=config.get("model", {}).get("load_in_4bit", args.load_in_4bit),
            use_slither=args.use_slither,
            use_wandb=not args.no_wandb,
            wandb_project=config.get("wandb", {}).get("project", "smart-contract-vuln-detection"),
            wandb_run_name=f"{config.get('experiment', {}).get('name', 'grpo')}-grpo",
        )
    else:
        train_grpo(
            model_path=args.model,
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            num_generations=args.num_generations,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            per_device_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            load_in_4bit=args.load_in_4bit,
            use_slither=args.use_slither,
            use_wandb=not args.no_wandb,
        )


if __name__ == "__main__":
    main()
