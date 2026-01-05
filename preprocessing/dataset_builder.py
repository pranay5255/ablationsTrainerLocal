"""
Dataset Builder Module
======================
Builds HuggingFace Datasets for different training stages.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Iterator
from dataclasses import dataclass
import logging

from datasets import Dataset, DatasetDict, Features, Value, Sequence

from .data_loader import DataLoader, DataSource, SolidityContract
from .formatters import (
    format_sft_example,
    format_sft_chat,
    format_sft_alpaca,
    format_dpo_example,
    format_dpo_trl,
    format_cpt_example,
    SFTExample,
    DPOExample,
    CPTExample,
)

logger = logging.getLogger(__name__)


@dataclass
class DataMixture:
    """
    Data mixture configuration as per PRD Section 4.1

    Stage 1 (CPT):
    - 50% General high-quality code
    - 30% Solidity code
    - 10% Audit reports
    - 10% Math/reasoning traces

    Stage 2 (SFT):
    - 40% Vulnerability-labeled Solidity
    - 25% Synthetic vulnerability pairs
    - 20% Clean non-vulnerable contracts
    - 10% General code instruction data
    - 5% Security documentation
    """
    general_code: float = 0.50
    solidity_code: float = 0.30
    audit_reports: float = 0.10
    math_reasoning: float = 0.10

    # SFT specific
    labeled_vulns: float = 0.40
    synthetic_vulns: float = 0.25
    clean_contracts: float = 0.20
    general_instructions: float = 0.10
    security_docs: float = 0.05


class DatasetBuilder:
    """
    Builds training datasets for each stage.

    Usage:
        builder = DatasetBuilder(data_dir="/data")

        # Build SFT dataset
        sft_dataset = builder.build_sft_dataset(num_samples=10000)

        # Build DPO dataset
        dpo_dataset = builder.build_dpo_dataset(num_samples=5000)

        # Build CPT dataset
        cpt_dataset = builder.build_cpt_dataset(num_tokens=2_000_000_000)
    """

    def __init__(
        self,
        data_dir: Union[str, Path] = "/data",
        output_dir: Optional[Union[str, Path]] = None,
        seed: int = 42
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else Path("output/processed")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.loader = DataLoader(data_dir)
        self.seed = seed
        random.seed(seed)

    def build_sft_dataset(
        self,
        num_samples: Optional[int] = None,
        mixture: Optional[DataMixture] = None,
        format_type: str = "chat",  # "chat" or "alpaca"
        train_split: float = 0.9,
        save_path: Optional[str] = None
    ) -> DatasetDict:
        """
        Build SFT dataset from available data sources.

        Args:
            num_samples: Target number of samples (None = all available)
            mixture: Data mixture ratios
            format_type: Output format ("chat" for TRL, "alpaca" for LLaMA-Factory)
            train_split: Train/validation split ratio
            save_path: Optional path to save dataset

        Returns:
            DatasetDict with train and validation splits
        """
        if mixture is None:
            mixture = DataMixture()

        examples = []

        # Load vulnerable contracts (SmartBugs + Kaggle)
        logger.info("Loading vulnerable contracts...")
        vuln_count = 0
        target_vulns = int((num_samples or 10000) * mixture.labeled_vulns)

        # SmartBugs
        for contract in self.loader.load_smartbugs():
            if vuln_count >= target_vulns:
                break
            sft_example = format_sft_example(
                source_code=contract.source_code,
                vulnerabilities=contract.vulnerabilities,
                is_vulnerable=True
            )
            if format_type == "chat":
                examples.append({"messages": format_sft_chat(sft_example)})
            else:
                examples.append(format_sft_alpaca(sft_example))
            vuln_count += 1

        # Kaggle
        if vuln_count < target_vulns:
            for contract in self.loader.load_kaggle_vulnerability():
                if vuln_count >= target_vulns:
                    break
                sft_example = format_sft_example(
                    source_code=contract.source_code,
                    vulnerabilities=contract.vulnerabilities,
                    is_vulnerable=True
                )
                if format_type == "chat":
                    examples.append({"messages": format_sft_chat(sft_example)})
                else:
                    examples.append(format_sft_alpaca(sft_example))
                vuln_count += 1

        # Load clean contracts from Zellic (subset)
        logger.info("Loading clean contracts...")
        clean_count = 0
        target_clean = int((num_samples or 10000) * mixture.clean_contracts)

        for contract in self.loader.load_zellic():
            if clean_count >= target_clean:
                break
            sft_example = format_sft_example(
                source_code=contract.source_code,
                vulnerabilities=[],
                is_vulnerable=False
            )
            if format_type == "chat":
                examples.append({"messages": format_sft_chat(sft_example)})
            else:
                examples.append(format_sft_alpaca(sft_example))
            clean_count += 1

        logger.info(f"Built SFT dataset with {len(examples)} examples")
        logger.info(f"  - Vulnerable: {vuln_count}")
        logger.info(f"  - Clean: {clean_count}")

        # Shuffle and split
        random.shuffle(examples)
        split_idx = int(len(examples) * train_split)

        train_examples = examples[:split_idx]
        val_examples = examples[split_idx:]

        # Create HuggingFace datasets
        train_dataset = Dataset.from_list(train_examples)
        val_dataset = Dataset.from_list(val_examples)

        dataset_dict = DatasetDict({
            "train": train_dataset,
            "validation": val_dataset
        })

        # Save if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            dataset_dict.save_to_disk(str(save_path))
            logger.info(f"Saved SFT dataset to {save_path}")

            # Also save as JSONL for inspection
            self._save_jsonl(train_examples, save_path / "train.jsonl")
            self._save_jsonl(val_examples, save_path / "val.jsonl")

        return dataset_dict

    def build_dpo_dataset(
        self,
        num_samples: Optional[int] = None,
        save_path: Optional[str] = None
    ) -> DatasetDict:
        """
        Build DPO dataset with preference pairs.

        Creates pairs from:
        1. Correct detection vs. missed detection (false negative)
        2. Correct detection vs. false positive
        3. Correct severity vs. wrong severity

        Args:
            num_samples: Target number of preference pairs
            save_path: Optional path to save dataset

        Returns:
            DatasetDict with train and validation splits
        """
        examples = []

        logger.info("Building DPO preference pairs...")

        # Load vulnerable contracts
        for contract in self.loader.load_smartbugs():
            if num_samples and len(examples) >= num_samples:
                break

            if not contract.vulnerabilities:
                continue

            # Create preference pair: correct detection vs. false negative
            correct_analysis = {"vulnerabilities": contract.vulnerabilities}
            incorrect_analysis = {"vulnerabilities": []}  # Missed vulnerability

            dpo_example = format_dpo_example(
                source_code=contract.source_code,
                correct_analysis=correct_analysis,
                incorrect_analysis=incorrect_analysis
            )
            examples.append(format_dpo_trl(dpo_example))

        # Load clean contracts to create false positive pairs
        for contract in self.loader.load_zellic():
            if num_samples and len(examples) >= num_samples:
                break

            # Create preference pair: no vulnerability vs. false positive
            correct_analysis = {"vulnerabilities": []}
            incorrect_analysis = {
                "vulnerabilities": [{
                    "type": "Reentrancy",
                    "swc_id": "SWC-107",
                    "cwe_id": "CWE-841",
                    "severity": "High",
                    "lines": "1-10",
                    "explanation": "False positive - no actual vulnerability.",
                    "remediation": "No action needed."
                }]
            }

            dpo_example = format_dpo_example(
                source_code=contract.source_code,
                correct_analysis=correct_analysis,
                incorrect_analysis=incorrect_analysis
            )
            examples.append(format_dpo_trl(dpo_example))

            # Limit clean contracts
            if len(examples) >= (num_samples or 1000) * 0.3:
                break

        logger.info(f"Built DPO dataset with {len(examples)} preference pairs")

        # Shuffle and split
        random.shuffle(examples)
        split_idx = int(len(examples) * 0.9)

        train_dataset = Dataset.from_list(examples[:split_idx])
        val_dataset = Dataset.from_list(examples[split_idx:])

        dataset_dict = DatasetDict({
            "train": train_dataset,
            "validation": val_dataset
        })

        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            dataset_dict.save_to_disk(str(save_path))
            logger.info(f"Saved DPO dataset to {save_path}")

        return dataset_dict

    def build_cpt_dataset(
        self,
        target_tokens: int = 2_000_000_000,  # 2B tokens
        mixture: Optional[DataMixture] = None,
        save_path: Optional[str] = None,
        streaming: bool = True
    ) -> Union[Dataset, Iterator[Dict[str, str]]]:
        """
        Build Continued Pretraining dataset.

        For large token counts, returns an iterator for streaming.

        Args:
            target_tokens: Target number of tokens
            mixture: Data mixture ratios
            save_path: Optional path to save (only for small datasets)
            streaming: Whether to return iterator (for large datasets)

        Returns:
            Dataset or Iterator of text examples
        """
        if mixture is None:
            mixture = DataMixture()

        def generate_examples():
            token_count = 0
            chars_per_token = 4  # Rough estimate

            # Solidity code (30%)
            solidity_target = int(target_tokens * mixture.solidity_code)
            for contract in self.loader.load_zellic():
                if token_count >= target_tokens:
                    break
                text = contract.source_code
                token_count += len(text) // chars_per_token
                yield {"text": text, "source": "solidity"}

            # Audit reports (10%)
            audit_target = int(target_tokens * mixture.audit_reports)
            for report in self.loader.load_audit_reports():
                if token_count >= target_tokens:
                    break
                text = report.get("content", "")
                if isinstance(text, dict):
                    text = json.dumps(text)
                token_count += len(text) // chars_per_token
                yield {"text": text, "source": "audit_report"}

        if streaming:
            return generate_examples()

        # For smaller datasets, collect all
        examples = list(generate_examples())
        dataset = Dataset.from_list(examples)

        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            dataset.save_to_disk(str(save_path))

        return dataset

    def build_grpo_prompts(
        self,
        num_prompts: int = 1000,
        save_path: Optional[str] = None
    ) -> Dataset:
        """
        Build prompt dataset for GRPO training.

        GRPO generates responses on-the-fly, so we only need prompts.

        Args:
            num_prompts: Number of prompts to generate
            save_path: Optional path to save dataset

        Returns:
            Dataset with prompts for GRPO
        """
        prompts = []
        instruction = "Analyze the following Solidity smart contract for security vulnerabilities."

        # Mix vulnerable and clean contracts
        for contract in self.loader.load_smartbugs():
            if len(prompts) >= num_prompts:
                break
            prompt = f"{instruction}\n\n```solidity\n{contract.source_code}\n```"
            prompts.append({
                "prompt": prompt,
                "has_vulnerability": True,
                "ground_truth": contract.vulnerabilities
            })

        for contract in self.loader.load_zellic():
            if len(prompts) >= num_prompts:
                break
            prompt = f"{instruction}\n\n```solidity\n{contract.source_code}\n```"
            prompts.append({
                "prompt": prompt,
                "has_vulnerability": False,
                "ground_truth": []
            })

        random.shuffle(prompts)
        dataset = Dataset.from_list(prompts[:num_prompts])

        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            dataset.save_to_disk(str(save_path))

        return dataset

    def _save_jsonl(self, examples: List[Dict], path: Path):
        """Save examples as JSONL"""
        with open(path, 'w') as f:
            for ex in examples:
                f.write(json.dumps(ex) + '\n')


def create_datasets_for_ablation(
    data_dir: str = "/data",
    output_dir: str = "output/processed",
    sft_samples: int = 10000,
    dpo_samples: int = 5000,
    grpo_prompts: int = 1000
) -> Dict[str, Path]:
    """
    Convenience function to create all datasets for ablation experiments.

    Args:
        data_dir: Path to raw data
        output_dir: Path for processed datasets
        sft_samples: Number of SFT examples
        dpo_samples: Number of DPO preference pairs
        grpo_prompts: Number of GRPO prompts

    Returns:
        Dictionary of dataset paths
    """
    builder = DatasetBuilder(data_dir=data_dir, output_dir=output_dir)
    output_dir = Path(output_dir)

    paths = {}

    # SFT dataset
    logger.info("Building SFT dataset...")
    builder.build_sft_dataset(
        num_samples=sft_samples,
        format_type="chat",
        save_path=output_dir / "sft"
    )
    paths["sft"] = output_dir / "sft"

    # DPO dataset
    logger.info("Building DPO dataset...")
    builder.build_dpo_dataset(
        num_samples=dpo_samples,
        save_path=output_dir / "dpo"
    )
    paths["dpo"] = output_dir / "dpo"

    # GRPO prompts
    logger.info("Building GRPO prompts...")
    builder.build_grpo_prompts(
        num_prompts=grpo_prompts,
        save_path=output_dir / "grpo"
    )
    paths["grpo"] = output_dir / "grpo"

    logger.info(f"All datasets created in {output_dir}")
    return paths
