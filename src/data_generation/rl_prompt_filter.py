import asyncio
import json
import logging
import re
from typing import List, Dict, Any, Optional
from .llm_client import LLMClient

logger = logging.getLogger(__name__)

class RLPromptFilter:
    def __init__(self, client: LLMClient, pass_rate_threshold: float = 0.625):
        self.client = client
        self.threshold = pass_rate_threshold

    def _extract_swc(self, text: str) -> Optional[str]:
        match = re.search(r"SWC[:\s]*([0-9]{3})", text, re.IGNORECASE)
        if match:
            return f"SWC-{match.group(1)}"
        return None

    def _verify_completion(self, completion: str, ground_truth: List[Dict[str, Any]]) -> float:
        """
        Simple SWC-based verifier for filtering.
        In production, this would use detection_verifier.py.
        """
        detected_swc = self._extract_swc(completion)
        if not detected_swc:
            return 0.0
        
        # Check if detected SWC matches any ground truth SWC
        gt_swcs = [v.get("swc_id") for v in ground_truth]
        if detected_swc in gt_swcs:
            return 1.0
        
        # If ground truth is empty and model says None or no SWC found
        if not gt_swcs and (detected_swc == "None" or "no vulnerability" in completion.lower()):
            return 1.0

        return 0.0

    async def filter_prompts(self, contracts: List[Dict[str, Any]], n_completions: int = 8) -> List[Dict[str, Any]]:
        filtered_prompts = []
        
        for contract in contracts:
            code = contract.get("source_code", "")
            gt = contract.get("vulnerabilities", [])
            prompt = f"Analyze the following Solidity smart contract for security vulnerabilities.\n\n```solidity\n{code}\n```"
            
            logger.info(f"Evaluating prompt difficulty for {contract.get('file_path')}...")
            
            # Generate N completions
            completions = []
            for _ in range(n_completions):
                comp = await self.client.generate(prompt)
                if comp:
                    completions.append(comp)
            
            if not completions:
                continue

            # Calculate pass rate
            rewards = [self._verify_completion(c, gt) for c in completions]
            pass_rate = sum(rewards) / len(rewards)
            
            logger.info(f"Pass rate: {pass_rate:.3f} (Threshold: {self.threshold})")
            
            if pass_rate <= self.threshold:
                filtered_prompts.append({
                    "prompt": prompt,
                    "ground_truth": gt,
                    "pass_rate": pass_rate,
                    "metadata": contract.get("metadata", {})
                })
                logger.info(f"Prompt kept: {contract.get('file_path')}")
            else:
                logger.info(f"Prompt filtered (too easy): {contract.get('file_path')}")

        return filtered_prompts

    async def process_dataset(self, contracts: List[Dict[str, Any]], output_file: str):
        filtered = await self.filter_prompts(contracts)
        with open(output_file, 'w') as f:
            json.dump(filtered, f, indent=2)
        return filtered

