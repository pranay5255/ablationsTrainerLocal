#!/usr/bin/env python3
"""
Ablation Experiment Orchestrator
================================
Runs ablation experiments on RTX 4090 as defined in experiment configs.

This script orchestrates the complete training pipeline:
1. Data preprocessing (THINK, INSTRUCT, RL-ZERO variants)
2. SFT training (detailed traces for THINK, concise for INSTRUCT)
3. DPO training (Delta Learning: Strong vs Weak)
4. RLVR training (OlmoRL: continuous batching, asymmetric clipping)
5. Evaluation on SmartBugs

Usage:
    # Run THINK ablation
    python run_ablations.py --config experiments/ablations/smollm2_135m.yaml --variant THINK

import os
import sys
import json
import argparse
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
ABLATIONS_DIR = EXPERIMENTS_DIR / "ablations"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
LOGS_DIR = PROJECT_ROOT / "logs"
OUTPUT_DIR = PROJECT_ROOT / "output"


def load_config(config_path: str) -> Dict[str, Any]:
    """Load experiment configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_all_ablation_configs() -> List[Path]:
    """Get all ablation configuration files."""
    return list(ABLATIONS_DIR.glob("*.yaml"))


def setup_directories():
    """Ensure all required directories exist."""
    for dir_path in [CHECKPOINTS_DIR, LOGS_DIR, OUTPUT_DIR, OUTPUT_DIR / "processed"]:
        dir_path.mkdir(parents=True, exist_ok=True)


def run_command(cmd: List[str], dry_run: bool = False) -> int:
    """Run a command and return exit code."""
    cmd_str = " ".join(cmd)
    logger.info(f"Running: {cmd_str}")

    if dry_run:
        logger.info("[DRY RUN] Would execute command")
        return 0

    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        return e.returncode


class AblationRunner:
    """
    Orchestrates ablation experiments.
    """

    def __init__(
        self,
        config_path: str,
        variant: str = "THINK",
        data_dir: str = "output",
        dry_run: bool = False,
        skip_preprocess: bool = False,
    ):
        self.config_path = Path(config_path)
        self.config = load_config(config_path)
        self.variant = variant
        self.data_dir = Path(data_dir)
        self.dry_run = dry_run
        self.skip_preprocess = skip_preprocess

        # Extract config values
        self.experiment_name = self.config.get("experiment", {}).get("name", "ablation")
        self.experiment_name = f"{self.experiment_name}-{self.variant.lower()}"
        self.model_name = self.config.get("model", {}).get("name")
        self.output_base = Path(self.config.get("output", {}).get("dir", f"checkpoints/{self.experiment_name}"))

        # Ensure directories exist
        self.output_base.mkdir(parents=True, exist_ok=True)

        # Log file for this run
        self.log_file = LOGS_DIR / f"{self.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    def preprocess_data(self) -> bool:
        """Run data preprocessing to create training datasets."""
        logger.info("=" * 60)
        logger.info("STAGE 0: Data Preprocessing")
        logger.info("=" * 60)

        if self.skip_preprocess:
            logger.info("Skipping preprocessing (--skip-preprocess)")
            return True

        # Check if processed data already exists
        sft_path = OUTPUT_DIR / "processed" / "sft"
        if sft_path.exists():
            logger.info(f"Processed data already exists at {sft_path}")
            return True

        # Run preprocessing
        cmd = [
            sys.executable,
            "-c",
            f"""
import sys
sys.path.insert(0, '{PROJECT_ROOT}')
from preprocessing.dataset_builder import create_datasets_for_ablation
create_datasets_for_ablation(
    data_dir='{self.data_dir}',
    output_dir='{OUTPUT_DIR / "processed"}',
    variant='{self.variant}',
    sft_samples=10000,
    dpo_samples=5000,
    rl_prompts=1000
)
"""
        ]

        return run_command(cmd, self.dry_run) == 0

    def run_sft(self) -> bool:
        """Run SFT training stage."""
        logger.info("=" * 60)
        logger.info("STAGE 1: Supervised Fine-Tuning (SFT)")
        logger.info("=" * 60)

        sft_config = self.config.get("stages", {}).get("sft", {})
        if not sft_config.get("enabled", True):
            logger.info("SFT stage disabled in config")
            return True

        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "training" / "scripts" / "train_sft.py"),
            "--config", str(self.config_path),
        ]

        if self.dry_run:
            cmd.append("--no-wandb")

        return run_command(cmd, self.dry_run) == 0

    def run_dpo(self) -> bool:
        """Run DPO training stage."""
        logger.info("=" * 60)
        logger.info("STAGE 2: Direct Preference Optimization (DPO)")
        logger.info("=" * 60)

        dpo_config = self.config.get("stages", {}).get("dpo", {})
        if not dpo_config.get("enabled", True):
            logger.info("DPO stage disabled in config")
            return True

        # Check if SFT checkpoint exists
        sft_checkpoint = self.output_base / "sft"
        if not sft_checkpoint.exists() and not self.dry_run:
            logger.warning(f"SFT checkpoint not found at {sft_checkpoint}")
            logger.warning("Using base model for DPO")

        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "training" / "scripts" / "train_dpo.py"),
            "--config", str(self.config_path),
        ]

        if self.dry_run:
            cmd.append("--no-wandb")

        return run_command(cmd, self.dry_run) == 0

    def run_rlvr(self) -> bool:
        """Run RLVR training stage."""
        logger.info("=" * 60)
        logger.info("STAGE 3: RL with Verifiable Rewards (RLVR)")
        logger.info("=" * 60)

        rlvr_config = self.config.get("stages", {}).get("rlvr", {})
        if not rlvr_config.get("enabled", False):
            logger.info("RLVR stage disabled in config")
            return True

        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "training" / "scripts" / "train_rlvr.py"),
            "--config", str(self.config_path),
            "--variant", self.variant,
        ]

        if self.dry_run:
            cmd.append("--no-wandb")

        return run_command(cmd, self.dry_run) == 0

    def run_evaluation(self) -> Dict[str, Any]:
        """Run evaluation on SmartBugs benchmark."""
        logger.info("=" * 60)
        logger.info("STAGE 4: Evaluation on SmartBugs")
        logger.info("=" * 60)

        # Find best checkpoint
        checkpoint_priorities = ["rlvr", "dpo", "sft"]
        checkpoint_path = None

        for stage in checkpoint_priorities:
            candidate = self.output_base / stage
            if candidate.exists():
                checkpoint_path = candidate
                break

        if checkpoint_path is None and not self.dry_run:
            logger.error("No checkpoint found for evaluation")
            return {"error": "No checkpoint found"}

        if checkpoint_path is None:
            checkpoint_path = self.output_base / "sft"  # Dummy for dry run

        # Run evaluation
        eval_output = self.output_base / "evaluation_results.json"

        cmd = [
            sys.executable,
            str(EXPERIMENTS_DIR / "evaluation" / "smartbugs_eval.py"),
            "--model", str(checkpoint_path),
            "--dataset", str(self.data_dir / "smartbugs"),
            "--output", str(eval_output),
        ]

        if self.config.get("model", {}).get("load_in_4bit", False):
            cmd.append("--load-in-4bit")

        exit_code = run_command(cmd, self.dry_run)

        # Load and return results
        if eval_output.exists() and not self.dry_run:
            with open(eval_output) as f:
                return json.load(f)

        return {"exit_code": exit_code}

    def run_full_pipeline(self, stages: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run the full ablation pipeline.

        Args:
            stages: Optional list of stages to run. If None, runs all enabled stages.

        Returns:
            Dictionary with results from each stage.
        """
        results = {
            "experiment_name": self.experiment_name,
            "model": self.model_name,
            "config_path": str(self.config_path),
            "start_time": datetime.now().isoformat(),
            "stages": {},
        }

        all_stages = ["preprocess", "sft", "dpo", "rlvr", "eval"]

        if stages is None:
            stages = all_stages

        stage_functions = {
            "preprocess": self.preprocess_data,
            "sft": self.run_sft,
            "dpo": self.run_dpo,
            "rlvr": self.run_rlvr,
            "eval": self.run_evaluation,
        }

        for stage in stages:
            if stage not in stage_functions:
                logger.warning(f"Unknown stage: {stage}")
                continue

            logger.info(f"\n{'#' * 60}")
            logger.info(f"# Running stage: {stage}")
            logger.info(f"{'#' * 60}\n")

            try:
                result = stage_functions[stage]()
                results["stages"][stage] = {
                    "success": result if isinstance(result, bool) else True,
                    "result": result if isinstance(result, dict) else None,
                }

                if isinstance(result, bool) and not result:
                    logger.error(f"Stage {stage} failed, stopping pipeline")
                    break

            except Exception as e:
                logger.error(f"Stage {stage} failed with error: {e}")
                results["stages"][stage] = {
                    "success": False,
                    "error": str(e),
                }
                break

        results["end_time"] = datetime.now().isoformat()

        # Save results
        results_path = self.output_base / "ablation_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {results_path}")

        return results


def run_all_ablations(dry_run: bool = False, data_dir: str = "/data"):
    """Run all ablation experiments sequentially."""
    configs = get_all_ablation_configs()
    logger.info(f"Found {len(configs)} ablation configurations")

    all_results = {}

    for config_path in configs:
        logger.info(f"\n{'=' * 70}")
        logger.info(f"Running ablation: {config_path.stem}")
        logger.info(f"{'=' * 70}\n")

        runner = AblationRunner(
            config_path=str(config_path),
            data_dir=data_dir,
            dry_run=dry_run,
            skip_preprocess=bool(all_results),  # Skip after first
        )

        results = runner.run_full_pipeline()
        all_results[config_path.stem] = results

    # Save combined results
    combined_path = LOGS_DIR / f"all_ablations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(combined_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\nAll ablation results saved to {combined_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("ABLATION SUMMARY")
    print("=" * 70)

    for name, results in all_results.items():
        eval_result = results.get("stages", {}).get("eval", {}).get("result", {})
        metrics = eval_result.get("metrics", {})

        f1 = metrics.get("f1", "N/A")
        precision = metrics.get("precision", "N/A")
        recall = metrics.get("recall", "N/A")

        if isinstance(f1, float):
            print(f"{name:30s} F1: {f1:.3f}  Precision: {precision:.3f}  Recall: {recall:.3f}")
        else:
            print(f"{name:30s} F1: {f1}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Run ablation experiments for smart contract vulnerability detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single ablation with config
  python run_ablations.py --config experiments/ablations/smollm2_135m.yaml

  # Run all ablations
  python run_ablations.py --all

  # Run only SFT stage
  python run_ablations.py --config experiments/ablations/smollm2_135m.yaml --stage sft

  # Run SFT and DPO stages
  python run_ablations.py --config experiments/ablations/smollm2_135m.yaml --stage sft --stage dpo

  # Dry run to see what would be executed
  python run_ablations.py --config experiments/ablations/smollm2_135m.yaml --dry-run
        """
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to experiment configuration YAML"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all ablation experiments"
    )
    parser.add_argument(
        "--variant",
        choices=["THINK", "INSTRUCT", "RL-ZERO"],
        default="THINK",
        help="Model variant to train (default: THINK)"
    )
    parser.add_argument(
        "--stage",
        action="append",
        choices=["preprocess", "sft", "dpo", "rlvr", "eval"],
        help="Run specific stage(s) only (can be specified multiple times)"
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip data preprocessing stage"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="output",
        help="Path to data directory"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing"
    )
    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="List available ablation configurations"
    )

    args = parser.parse_args()

    # Setup directories
    setup_directories()

    if args.list_configs:
        configs = get_all_ablation_configs()
        print("\nAvailable ablation configurations:")
        for config in configs:
            print(f"  - {config}")
        return

    if args.all:
        run_all_ablations(dry_run=args.dry_run, data_dir=args.data_dir)
        return

    if not args.config:
        parser.print_help()
        print("\nError: Please specify --config or --all")
        sys.exit(1)

    # Run single ablation
    runner = AblationRunner(
        config_path=args.config,
        variant=args.variant,
        data_dir=args.data_dir,
        dry_run=args.dry_run,
        skip_preprocess=args.skip_preprocess,
    )

    results = runner.run_full_pipeline(stages=args.stage)

    # Print final status
    print("\n" + "=" * 60)
    print("ABLATION COMPLETE")
    print("=" * 60)

    for stage, stage_result in results.get("stages", {}).items():
        status = "PASS" if stage_result.get("success") else "FAIL"
        print(f"  {stage:15s} [{status}]")

    # Check evaluation results
    eval_result = results.get("stages", {}).get("eval", {}).get("result", {})
    metrics = eval_result.get("metrics", {})

    if metrics:
        print(f"\nFinal Metrics:")
        print(f"  Precision: {metrics.get('precision', 'N/A'):.3f}")
        print(f"  Recall:    {metrics.get('recall', 'N/A'):.3f}")
        print(f"  F1:        {metrics.get('f1', 'N/A'):.3f}")

        # Check against targets
        target_f1 = 0.45
        if metrics.get('f1', 0) >= target_f1:
            print(f"\n  Result: PASSED (F1 >= {target_f1})")
        else:
            print(f"\n  Result: NEEDS IMPROVEMENT (F1 < {target_f1})")

    print("=" * 60)


if __name__ == "__main__":
    main()
