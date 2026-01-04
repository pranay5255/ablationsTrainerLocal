"""
Data Formatters Module
======================
Format data for different training stages (CPT, SFT, DPO, GRPO).
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass


# =============================================================================
# Output Format Templates (as per PRD Section 3.1)
# =============================================================================

VULNERABILITY_OUTPUT_TEMPLATE = """Vulnerability: {vuln_type}
CWE: {cwe_id}
SWC: {swc_id}
Lines: {line_range}
Severity: {severity}
Explanation: {explanation}
Remediation: {remediation}"""

NO_VULNERABILITY_OUTPUT = """Vulnerability: None
Explanation: No vulnerabilities detected in this contract. The code follows security best practices."""


# =============================================================================
# SFT Formatters
# =============================================================================

@dataclass
class SFTExample:
    """Supervised Fine-Tuning example"""
    instruction: str
    input: str
    output: str
    metadata: Optional[Dict[str, Any]] = None


def format_sft_example(
    source_code: str,
    vulnerabilities: List[Dict[str, Any]],
    is_vulnerable: bool = True,
    include_system_prompt: bool = True
) -> SFTExample:
    """
    Format a contract for SFT training.

    Args:
        source_code: Solidity source code
        vulnerabilities: List of vulnerability annotations
        is_vulnerable: Whether the contract is vulnerable
        include_system_prompt: Whether to include system context

    Returns:
        SFTExample ready for training
    """
    system_prompt = """You are a smart contract security auditor. Analyze Solidity code for vulnerabilities.
For each vulnerability found, provide:
- Vulnerability type and SWC/CWE identifiers
- Affected line numbers
- Severity level (Critical/High/Medium/Low/Informational)
- Detailed explanation of the vulnerability
- Recommended remediation steps"""

    instruction = "Analyze the following Solidity smart contract for security vulnerabilities."

    if include_system_prompt:
        instruction = f"{system_prompt}\n\n{instruction}"

    if is_vulnerable and vulnerabilities:
        # Format each vulnerability
        outputs = []
        for vuln in vulnerabilities:
            output = VULNERABILITY_OUTPUT_TEMPLATE.format(
                vuln_type=vuln.get("type", vuln.get("name", "Unknown")),
                cwe_id=vuln.get("cwe_id", "CWE-Unknown"),
                swc_id=vuln.get("swc_id", "SWC-Unknown"),
                line_range=vuln.get("lines", vuln.get("line_range", "Unknown")),
                severity=vuln.get("severity", "Medium"),
                explanation=vuln.get("explanation", vuln.get("description", "Vulnerability detected.")),
                remediation=vuln.get("remediation", "Review and fix the vulnerable code pattern.")
            )
            outputs.append(output)
        output_text = "\n\n".join(outputs)
    else:
        output_text = NO_VULNERABILITY_OUTPUT

    return SFTExample(
        instruction=instruction,
        input=source_code,
        output=output_text,
        metadata={"is_vulnerable": is_vulnerable, "num_vulns": len(vulnerabilities) if vulnerabilities else 0}
    )


def format_sft_chat(example: SFTExample) -> List[Dict[str, str]]:
    """Convert SFT example to chat format for TRL"""
    return [
        {"role": "user", "content": f"{example.instruction}\n\n```solidity\n{example.input}\n```"},
        {"role": "assistant", "content": example.output}
    ]


def format_sft_alpaca(example: SFTExample) -> Dict[str, str]:
    """Convert SFT example to Alpaca format"""
    return {
        "instruction": example.instruction,
        "input": example.input,
        "output": example.output
    }


# =============================================================================
# DPO Formatters
# =============================================================================

@dataclass
class DPOExample:
    """Direct Preference Optimization example"""
    prompt: str
    chosen: str
    rejected: str
    metadata: Optional[Dict[str, Any]] = None


def format_dpo_example(
    source_code: str,
    correct_analysis: Dict[str, Any],
    incorrect_analysis: Dict[str, Any],
    instruction: Optional[str] = None
) -> DPOExample:
    """
    Format a contract for DPO training.

    Creates preference pairs where:
    - chosen: Correct vulnerability detection
    - rejected: Incorrect detection (false positive/negative, wrong type, etc.)

    Args:
        source_code: Solidity source code
        correct_analysis: The correct vulnerability analysis
        incorrect_analysis: The incorrect analysis (for contrast)
        instruction: Optional custom instruction

    Returns:
        DPOExample ready for training
    """
    if instruction is None:
        instruction = "Analyze the following Solidity smart contract for security vulnerabilities."

    prompt = f"{instruction}\n\n```solidity\n{source_code}\n```"

    # Format chosen response (correct)
    if correct_analysis.get("vulnerabilities"):
        chosen_parts = []
        for vuln in correct_analysis["vulnerabilities"]:
            chosen_parts.append(VULNERABILITY_OUTPUT_TEMPLATE.format(
                vuln_type=vuln.get("type", "Unknown"),
                cwe_id=vuln.get("cwe_id", "CWE-Unknown"),
                swc_id=vuln.get("swc_id", "SWC-Unknown"),
                line_range=vuln.get("lines", "Unknown"),
                severity=vuln.get("severity", "Medium"),
                explanation=vuln.get("explanation", ""),
                remediation=vuln.get("remediation", "")
            ))
        chosen = "\n\n".join(chosen_parts)
    else:
        chosen = NO_VULNERABILITY_OUTPUT

    # Format rejected response (incorrect)
    if incorrect_analysis.get("vulnerabilities"):
        rejected_parts = []
        for vuln in incorrect_analysis["vulnerabilities"]:
            rejected_parts.append(VULNERABILITY_OUTPUT_TEMPLATE.format(
                vuln_type=vuln.get("type", "Unknown"),
                cwe_id=vuln.get("cwe_id", "CWE-Unknown"),
                swc_id=vuln.get("swc_id", "SWC-Unknown"),
                line_range=vuln.get("lines", "Unknown"),
                severity=vuln.get("severity", "Medium"),
                explanation=vuln.get("explanation", ""),
                remediation=vuln.get("remediation", "")
            ))
        rejected = "\n\n".join(rejected_parts)
    else:
        rejected = NO_VULNERABILITY_OUTPUT

    return DPOExample(
        prompt=prompt,
        chosen=chosen,
        rejected=rejected,
        metadata={
            "correct_vuln_count": len(correct_analysis.get("vulnerabilities", [])),
            "incorrect_vuln_count": len(incorrect_analysis.get("vulnerabilities", []))
        }
    )


def format_dpo_trl(example: DPOExample) -> Dict[str, str]:
    """Convert DPO example to TRL DPOTrainer format"""
    return {
        "prompt": example.prompt,
        "chosen": example.chosen,
        "rejected": example.rejected
    }


# =============================================================================
# GRPO Formatters
# =============================================================================

@dataclass
class GRPOExample:
    """Group Relative Policy Optimization example"""
    prompt: str
    responses: List[str]
    rewards: List[float]
    metadata: Optional[Dict[str, Any]] = None


def format_grpo_example(
    source_code: str,
    model_responses: List[str],
    reward_scores: List[float],
    instruction: Optional[str] = None
) -> GRPOExample:
    """
    Format for GRPO training.

    GRPO uses multiple responses per prompt with relative rankings.

    Args:
        source_code: Solidity source code
        model_responses: List of model-generated responses
        reward_scores: Corresponding reward scores from reward function
        instruction: Optional custom instruction

    Returns:
        GRPOExample ready for training
    """
    if instruction is None:
        instruction = "Analyze the following Solidity smart contract for security vulnerabilities."

    prompt = f"{instruction}\n\n```solidity\n{source_code}\n```"

    return GRPOExample(
        prompt=prompt,
        responses=model_responses,
        rewards=reward_scores,
        metadata={"num_responses": len(model_responses)}
    )


# =============================================================================
# CPT Formatters
# =============================================================================

@dataclass
class CPTExample:
    """Continued Pretraining example"""
    text: str
    source: str
    metadata: Optional[Dict[str, Any]] = None


def format_cpt_example(
    source_code: str,
    source_name: str = "solidity",
    include_header: bool = False
) -> CPTExample:
    """
    Format for Continued Pretraining.

    CPT uses raw text without instruction formatting.

    Args:
        source_code: Raw source code or text
        source_name: Name of the data source
        include_header: Whether to include source header

    Returns:
        CPTExample ready for training
    """
    if include_header:
        text = f"// Source: {source_name}\n{source_code}"
    else:
        text = source_code

    return CPTExample(
        text=text,
        source=source_name,
        metadata={"length": len(text)}
    )


def format_cpt_audit_report(
    report_content: str,
    source: str = "audit_report"
) -> CPTExample:
    """Format an audit report for CPT"""
    return CPTExample(
        text=report_content,
        source=source,
        metadata={"type": "audit_report"}
    )


# =============================================================================
# Tokenization Helpers
# =============================================================================

def truncate_to_max_length(
    text: str,
    max_length: int = 2048,
    tokenizer=None
) -> str:
    """
    Truncate text to fit within max token length.

    If tokenizer provided, truncates by tokens. Otherwise by characters (rough estimate).
    """
    if tokenizer is not None:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
            text = tokenizer.decode(tokens)
    else:
        # Rough estimate: 4 chars per token
        char_limit = max_length * 4
        if len(text) > char_limit:
            text = text[:char_limit]

    return text


def create_chat_template(
    messages: List[Dict[str, str]],
    tokenizer=None,
    add_generation_prompt: bool = False
) -> str:
    """
    Apply chat template to messages.

    Falls back to simple formatting if tokenizer doesn't have chat template.
    """
    if tokenizer is not None and hasattr(tokenizer, 'apply_chat_template'):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt
        )

    # Fallback formatting
    formatted = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            formatted += f"<|system|>\n{content}\n"
        elif role == "user":
            formatted += f"<|user|>\n{content}\n"
        elif role == "assistant":
            formatted += f"<|assistant|>\n{content}\n"

    if add_generation_prompt:
        formatted += "<|assistant|>\n"

    return formatted
