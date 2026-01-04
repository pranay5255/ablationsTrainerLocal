#!/usr/bin/env python3
"""
SmartBugs Evaluation Harness
============================
Evaluates vulnerability detection models on the SmartBugs-curated benchmark.

Metrics (as per PRD Section 6.1):
- Precision: >35% target
- Recall: >70% target
- F1-score: >0.45 macro target
- False positive rate: <65% target

Usage:
    python smartbugs_eval.py --model checkpoints/sft --dataset /data/smartbugs
    python smartbugs_eval.py --config experiments/ablations/smollm2_135m.yaml
"""

import os
import sys
import json
import re
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Evaluation Data Structures
# =============================================================================

@dataclass
class PredictedVulnerability:
    """Model's predicted vulnerability"""
    swc_id: str
    cwe_id: Optional[str] = None
    vuln_type: str = ""
    severity: str = "Medium"
    lines: Optional[str] = None
    explanation: str = ""
    confidence: float = 1.0


@dataclass
class GroundTruthVulnerability:
    """Ground truth vulnerability annotation"""
    swc_id: str
    cwe_id: Optional[str] = None
    vuln_type: str = ""
    lines: Optional[str] = None


@dataclass
class EvaluationResult:
    """Evaluation metrics for a single contract"""
    contract_name: str
    has_ground_truth_vulns: bool
    predicted_vulns: List[PredictedVulnerability]
    ground_truth_vulns: List[GroundTruthVulnerability]
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    matched_swcs: List[str] = field(default_factory=list)


@dataclass
class AggregateMetrics:
    """Aggregate evaluation metrics"""
    total_contracts: int = 0
    vulnerable_contracts: int = 0
    clean_contracts: int = 0

    total_true_positives: int = 0
    total_false_positives: int = 0
    total_false_negatives: int = 0
    total_true_negatives: int = 0

    # Per-SWC metrics
    swc_metrics: Dict[str, Dict[str, int]] = field(default_factory=dict)

    @property
    def precision(self) -> float:
        """Calculate precision"""
        denom = self.total_true_positives + self.total_false_positives
        return self.total_true_positives / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        """Calculate recall"""
        denom = self.total_true_positives + self.total_false_negatives
        return self.total_true_positives / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        """Calculate F1 score"""
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)

    @property
    def false_positive_rate(self) -> float:
        """Calculate false positive rate"""
        denom = self.total_false_positives + self.total_true_negatives
        return self.total_false_positives / denom if denom > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "total_contracts": self.total_contracts,
            "vulnerable_contracts": self.vulnerable_contracts,
            "clean_contracts": self.clean_contracts,
            "true_positives": self.total_true_positives,
            "false_positives": self.total_false_positives,
            "false_negatives": self.total_false_negatives,
            "true_negatives": self.total_true_negatives,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "false_positive_rate": self.false_positive_rate,
            "swc_metrics": self.swc_metrics,
        }


# =============================================================================
# Parsing Functions
# =============================================================================

def parse_model_output(output: str) -> List[PredictedVulnerability]:
    """
    Parse model output to extract predicted vulnerabilities.

    Expected format (PRD Section 3.1):
    Vulnerability: Reentrancy
    CWE: CWE-841
    SWC: SWC-107
    Lines: 12-15
    Severity: High
    Explanation: ...
    """
    vulnerabilities = []

    # Check for "no vulnerability" responses
    if re.search(r'vulnerability[:\s]*none|no vulnerabilit', output.lower()):
        return vulnerabilities

    # Split by vulnerability blocks
    vuln_blocks = re.split(r'\n\s*\n+', output)

    for block in vuln_blocks:
        if not block.strip():
            continue

        vuln = PredictedVulnerability(swc_id="")

        # Extract SWC
        swc_match = re.search(r'SWC[:\s]*(SWC-?\d+|\d+)', block, re.IGNORECASE)
        if swc_match:
            swc = swc_match.group(1)
            if not swc.startswith('SWC'):
                swc = f"SWC-{swc}"
            vuln.swc_id = swc.upper()

        # Extract CWE
        cwe_match = re.search(r'CWE[:\s]*(CWE-?\d+|\d+)', block, re.IGNORECASE)
        if cwe_match:
            cwe = cwe_match.group(1)
            if not cwe.startswith('CWE'):
                cwe = f"CWE-{cwe}"
            vuln.cwe_id = cwe.upper()

        # Extract vulnerability type
        type_match = re.search(r'Vulnerability[:\s]*([^\n]+)', block, re.IGNORECASE)
        if type_match:
            vuln.vuln_type = type_match.group(1).strip()

        # Extract lines
        lines_match = re.search(r'Lines?[:\s]*(\d+(?:\s*-\s*\d+)?)', block, re.IGNORECASE)
        if lines_match:
            vuln.lines = lines_match.group(1).strip()

        # Extract severity
        severity_match = re.search(r'Severity[:\s]*(\w+)', block, re.IGNORECASE)
        if severity_match:
            vuln.severity = severity_match.group(1).strip()

        # Extract explanation
        expl_match = re.search(r'Explanation[:\s]*([^\n]+)', block, re.IGNORECASE)
        if expl_match:
            vuln.explanation = expl_match.group(1).strip()

        # Only add if we found at least an SWC or vulnerability type
        if vuln.swc_id or vuln.vuln_type:
            if not vuln.swc_id:
                # Try to infer SWC from type
                vuln.swc_id = infer_swc_from_type(vuln.vuln_type)
            vulnerabilities.append(vuln)

    return vulnerabilities


def infer_swc_from_type(vuln_type: str) -> str:
    """Infer SWC ID from vulnerability type name"""
    type_lower = vuln_type.lower()

    mapping = {
        "reentrancy": "SWC-107",
        "reentrant": "SWC-107",
        "integer overflow": "SWC-101",
        "integer underflow": "SWC-101",
        "overflow": "SWC-101",
        "underflow": "SWC-101",
        "access control": "SWC-115",
        "authorization": "SWC-115",
        "unchecked": "SWC-104",
        "call": "SWC-104",
        "front-running": "SWC-114",
        "frontrunning": "SWC-114",
        "front running": "SWC-114",
        "timestamp": "SWC-116",
        "block.timestamp": "SWC-116",
        "randomness": "SWC-120",
        "random": "SWC-120",
        "denial": "SWC-128",
        "dos": "SWC-128",
        "tx.origin": "SWC-115",
    }

    for key, swc in mapping.items():
        if key in type_lower:
            return swc

    return "SWC-000"  # Unknown


# =============================================================================
# Evaluation Logic
# =============================================================================

def match_vulnerabilities(
    predicted: List[PredictedVulnerability],
    ground_truth: List[GroundTruthVulnerability],
    strict_swc: bool = True
) -> Tuple[int, int, int]:
    """
    Match predicted vulnerabilities against ground truth.

    Args:
        predicted: List of predicted vulnerabilities
        ground_truth: List of ground truth vulnerabilities
        strict_swc: If True, require exact SWC match

    Returns:
        Tuple of (true_positives, false_positives, false_negatives)
    """
    gt_swcs = set(gt.swc_id for gt in ground_truth)
    pred_swcs = set(p.swc_id for p in predicted)

    if strict_swc:
        # Exact SWC matching
        true_positives = len(gt_swcs & pred_swcs)
        false_positives = len(pred_swcs - gt_swcs)
        false_negatives = len(gt_swcs - pred_swcs)
    else:
        # Category-level matching (same first digit)
        gt_categories = set(swc[:7] if len(swc) >= 7 else swc for swc in gt_swcs)
        pred_categories = set(swc[:7] if len(swc) >= 7 else swc for swc in pred_swcs)

        true_positives = len(gt_categories & pred_categories)
        false_positives = len(pred_categories - gt_categories)
        false_negatives = len(gt_categories - pred_categories)

    return true_positives, false_positives, false_negatives


class SmartBugsEvaluator:
    """
    Evaluator for SmartBugs-curated benchmark.
    """

    def __init__(
        self,
        model_path: str,
        dataset_path: str = "/data/smartbugs",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        load_in_4bit: bool = False,
        max_seq_length: int = 2048,
    ):
        self.model_path = model_path
        self.dataset_path = Path(dataset_path)
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.max_seq_length = max_seq_length

        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load model and tokenizer"""
        logger.info(f"Loading model from {self.model_path}")

        try:
            from unsloth import FastLanguageModel
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_path,
                max_seq_length=self.max_seq_length,
                load_in_4bit=self.load_in_4bit,
            )
            FastLanguageModel.for_inference(self.model)
        except ImportError:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )

            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.bfloat16,
                "device_map": "auto",
            }

            if self.load_in_4bit:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )

            # Check for PEFT model
            if (Path(self.model_path) / "adapter_config.json").exists():
                from peft import AutoPeftModelForCausalLM
                self.model = AutoPeftModelForCausalLM.from_pretrained(
                    self.model_path, **model_kwargs
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path, **model_kwargs
                )

            self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("Model loaded successfully")

    def load_smartbugs_dataset(self) -> List[Dict[str, Any]]:
        """Load SmartBugs-curated contracts with annotations"""
        contracts = []

        if not self.dataset_path.exists():
            logger.warning(f"SmartBugs dataset not found at {self.dataset_path}")
            return contracts

        # Load from directory structure
        dataset_dir = self.dataset_path / "dataset"
        if dataset_dir.exists():
            for vuln_type_dir in dataset_dir.iterdir():
                if vuln_type_dir.is_dir():
                    vuln_type = vuln_type_dir.name
                    swc_id = infer_swc_from_type(vuln_type)

                    for sol_file in vuln_type_dir.glob("*.sol"):
                        with open(sol_file) as f:
                            source_code = f.read()

                        contracts.append({
                            "name": sol_file.name,
                            "source_code": source_code,
                            "vulnerabilities": [{
                                "swc_id": swc_id,
                                "type": vuln_type,
                            }],
                            "is_vulnerable": True,
                        })

        logger.info(f"Loaded {len(contracts)} contracts from SmartBugs")
        return contracts

    def generate_prediction(self, source_code: str) -> str:
        """Generate vulnerability analysis for a contract"""
        instruction = "Analyze the following Solidity smart contract for security vulnerabilities."
        prompt = f"{instruction}\n\n```solidity\n{source_code}\n```"

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_length - 512,  # Leave room for generation
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,  # Deterministic for evaluation
                temperature=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the generated part
        response = response[len(prompt):].strip()

        return response

    def evaluate(
        self,
        contracts: Optional[List[Dict]] = None,
        output_path: Optional[str] = None,
    ) -> AggregateMetrics:
        """
        Run evaluation on SmartBugs dataset.

        Args:
            contracts: Optional list of contracts (loads from dataset if None)
            output_path: Optional path to save detailed results

        Returns:
            AggregateMetrics with evaluation results
        """
        if self.model is None:
            self.load_model()

        if contracts is None:
            contracts = self.load_smartbugs_dataset()

        if not contracts:
            logger.error("No contracts to evaluate")
            return AggregateMetrics()

        metrics = AggregateMetrics()
        results = []

        logger.info(f"Evaluating on {len(contracts)} contracts...")

        for contract in tqdm(contracts, desc="Evaluating"):
            # Generate prediction
            prediction = self.generate_prediction(contract["source_code"])

            # Parse prediction
            predicted_vulns = parse_model_output(prediction)

            # Parse ground truth
            gt_vulns = [
                GroundTruthVulnerability(
                    swc_id=v.get("swc_id", "SWC-000"),
                    vuln_type=v.get("type", ""),
                )
                for v in contract.get("vulnerabilities", [])
            ]

            # Match vulnerabilities
            tp, fp, fn = match_vulnerabilities(predicted_vulns, gt_vulns)

            # Update metrics
            metrics.total_contracts += 1
            if contract.get("is_vulnerable", True):
                metrics.vulnerable_contracts += 1
            else:
                metrics.clean_contracts += 1
                if not predicted_vulns:
                    metrics.total_true_negatives += 1

            metrics.total_true_positives += tp
            metrics.total_false_positives += fp
            metrics.total_false_negatives += fn

            # Track per-SWC metrics
            for gt in gt_vulns:
                swc = gt.swc_id
                if swc not in metrics.swc_metrics:
                    metrics.swc_metrics[swc] = {"tp": 0, "fp": 0, "fn": 0}

                if swc in [p.swc_id for p in predicted_vulns]:
                    metrics.swc_metrics[swc]["tp"] += 1
                else:
                    metrics.swc_metrics[swc]["fn"] += 1

            # Store result
            results.append({
                "contract": contract["name"],
                "prediction": prediction,
                "predicted_vulns": [vars(v) for v in predicted_vulns],
                "ground_truth": [vars(v) for v in gt_vulns],
                "tp": tp,
                "fp": fp,
                "fn": fn,
            })

        # Save detailed results
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                json.dump({
                    "metrics": metrics.to_dict(),
                    "results": results,
                }, f, indent=2)

            logger.info(f"Detailed results saved to {output_path}")

        return metrics


def print_evaluation_report(metrics: AggregateMetrics, targets: Optional[Dict] = None):
    """Print formatted evaluation report"""
    if targets is None:
        targets = {
            "precision": 0.35,
            "recall": 0.70,
            "f1": 0.45,
            "false_positive_rate": 0.65,
        }

    print("\n" + "=" * 60)
    print("SMART CONTRACT VULNERABILITY DETECTION - EVALUATION REPORT")
    print("=" * 60)

    print(f"\nDataset Statistics:")
    print(f"  Total contracts:      {metrics.total_contracts}")
    print(f"  Vulnerable:           {metrics.vulnerable_contracts}")
    print(f"  Clean:                {metrics.clean_contracts}")

    print(f"\nDetection Results:")
    print(f"  True Positives:       {metrics.total_true_positives}")
    print(f"  False Positives:      {metrics.total_false_positives}")
    print(f"  False Negatives:      {metrics.total_false_negatives}")
    print(f"  True Negatives:       {metrics.total_true_negatives}")

    print(f"\nMetrics vs Targets:")

    def format_metric(name: str, value: float, target: float, higher_better: bool = True):
        pct = value * 100
        target_pct = target * 100
        if higher_better:
            status = "PASS" if value >= target else "FAIL"
        else:
            status = "PASS" if value <= target else "FAIL"
        return f"  {name:20s} {pct:6.1f}% (target: {target_pct:5.1f}%) [{status}]"

    print(format_metric("Precision:", metrics.precision, targets["precision"]))
    print(format_metric("Recall:", metrics.recall, targets["recall"]))
    print(format_metric("F1 Score:", metrics.f1, targets["f1"]))
    print(format_metric("False Positive Rate:", metrics.false_positive_rate,
                        targets["false_positive_rate"], higher_better=False))

    # Per-SWC breakdown
    if metrics.swc_metrics:
        print(f"\nPer-SWC Breakdown:")
        for swc, swc_m in sorted(metrics.swc_metrics.items()):
            tp, fn = swc_m["tp"], swc_m["fn"]
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            print(f"  {swc}: Recall = {recall:.1%} ({tp}/{tp+fn})")

    print("\n" + "=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="SmartBugs Evaluation Harness")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="/data/smartbugs",
        help="Path to SmartBugs dataset"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Output path for detailed results"
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Use 4-bit quantization"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Optional experiment config YAML"
    )

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = SmartBugsEvaluator(
        model_path=args.model,
        dataset_path=args.dataset,
        load_in_4bit=args.load_in_4bit,
    )

    # Run evaluation
    metrics = evaluator.evaluate(output_path=args.output)

    # Print report
    print_evaluation_report(metrics)

    # Return exit code based on F1 target
    target_f1 = 0.45
    if metrics.f1 >= target_f1:
        logger.info(f"Evaluation PASSED: F1 {metrics.f1:.3f} >= {target_f1}")
        sys.exit(0)
    else:
        logger.warning(f"Evaluation FAILED: F1 {metrics.f1:.3f} < {target_f1}")
        sys.exit(1)


if __name__ == "__main__":
    main()
