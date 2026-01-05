import asyncio
import json
import logging
from typing import List, Dict, Any
from .llm_client import LLMClient

logger = logging.getLogger(__name__)

class DeltaPairCreator:
    def __init__(self, strong_client: LLMClient, weak_client: LLMClient):
        self.strong_client = strong_client
        self.weak_client = weak_client

    async def create_pair(self, contract_code: str, instruction: str = "Analyze the following Solidity smart contract for security vulnerabilities.") -> Dict[str, str]:
        prompt = f"{instruction}\n\n```solidity\n{contract_code}\n```"
        
        logger.info("Generating responses for DPO Delta Learning...")
        # Generate in parallel
        strong_task = self.strong_client.generate(prompt)
        weak_task = self.weak_client.generate(prompt)
        
        strong_resp, weak_resp = await asyncio.gather(strong_task, weak_task)
        
        if not strong_resp or not weak_resp:
            return {}

        return {
            "prompt": prompt,
            "chosen": strong_resp,
            "rejected": weak_resp,
            "metadata": {
                "strong_model": self.strong_client.config.model,
                "weak_model": self.weak_client.config.model
            }
        }

    async def process_dataset(self, contracts: List[Dict[str, Any]], output_file: str):
        results = []
        for contract in contracts:
            code = contract.get("source_code", "")
            if not code:
                continue

            pair = await self.create_pair(code)
            if pair:
                # Length filtering as per PRD 6.3
                len_chosen = len(pair["chosen"].split())
                len_rejected = len(pair["rejected"].split())
                if abs(len_chosen - len_rejected) <= 100:
                    results.append(pair)
                    logger.info(f"Added DPO pair for {contract.get('file_path')}")
                else:
                    logger.warning(f"Filtered DPO pair due to length delta: {len_chosen} vs {len_rejected}")

            # Intermediate save
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
        
        return results
