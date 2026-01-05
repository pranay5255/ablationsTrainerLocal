import asyncio
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from .llm_client import LLMClient, LLMConfig

logger = logging.getLogger(__name__)

THINK_GEN_PROMPT = """Analyze this Solidity contract for vulnerabilities. Think through your analysis step by step:

1. First, identify the contract's purpose and key functions
2. Trace the flow of funds and state changes
3. Check for common vulnerability patterns (reentrancy, overflow, access control, etc.)
4. For each potential issue, explain WHY it's a vulnerability
5. Provide specific line numbers and evidence

Contract:
{contract_code}

Known vulnerability (if any): {ground_truth}

Output your analysis in this format:
<think>
[Your detailed reasoning process - be thorough, explain your logic]
</think>

Vulnerability: [Type]
SWC: [ID]
Lines: [X-Y]
Severity: [High/Medium/Low]
Explanation: [Concise summary]
Remediation: [Fix suggestion]"""

class TraceGenerator:
    def __init__(self, client: LLMClient):
        self.client = client

    async def generate_trace(self, contract_code: str, ground_truth: str) -> str:
        prompt = THINK_GEN_PROMPT.format(
            contract_code=contract_code,
            ground_truth=ground_truth
        )
        return await self.client.generate(prompt, max_tokens=16384)

    async def process_dataset(self, contracts: List[Dict[str, Any]], output_file: str, traces_per_contract: int = 1):
        """
        Process a list of contracts and generate reasoning traces.
        """
        results = []
        for contract in contracts:
            code = contract.get("source_code", "")
            vuln_info = json.dumps(contract.get("vulnerabilities", []))
            
            for i in range(traces_per_contract):
                logger.info(f"Generating trace {i+1}/{traces_per_contract} for {contract.get('file_path')}")
                trace = await self.generate_trace(code, vuln_info)
                if trace and "<think>" in trace:
                    results.append({
                        "instruction": "Analyze the following Solidity smart contract for security vulnerabilities.",
                        "input": code,
                        "output": trace,
                        "metadata": {
                            "source": contract.get("metadata", {}).get("source", "unknown"),
                            "file_path": contract.get("file_path"),
                            "trace_id": i
                        }
                    })
                
                # Immediate save to avoid data loss
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)

        return results
