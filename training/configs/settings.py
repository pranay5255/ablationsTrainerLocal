"""
Training Configuration Settings
================================
Central configuration for smart contract vulnerability detection training.
Optimized for RTX 4090 (24GB VRAM) single-GPU training.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum


# =============================================================================
# Path Configuration
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = Path("output")  # External data directory
OUTPUT_DIR = PROJECT_ROOT / "output"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
LOGS_DIR = PROJECT_ROOT / "logs"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# Create directories if they don't exist
for dir_path in [OUTPUT_DIR, CHECKPOINTS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Model Configuration
# =============================================================================
class ModelSize(Enum):
    """Ablation model sizes as per PRD Section 3.3"""
    SMOLLM2_135M = "HuggingFaceTB/SmolLM2-135M"
    SMOLLM2_360M = "HuggingFaceTB/SmolLM2-360M"
    QWEN_05B = "Qwen/Qwen2.5-0.5B"
    BAGUETTOTRON_321M = "PleIAs/Baguettotron-321M"
    SMOLLM2_17B = "HuggingFaceTB/SmolLM2-1.7B"
    QWEN_CODER_15B = "Qwen/Qwen2.5-Coder-1.5B"
    QWEN_CODER_3B = "Qwen/Qwen2.5-Coder-3B"  # Final model


@dataclass
class ModelConfig:
    """Model-specific configuration"""
    name: str
    model_id: str
    max_seq_length: int = 2048
    dtype: str = "bfloat16"  # RTX 4090 supports bf16
    load_in_4bit: bool = False  # Enable for larger models
    load_in_8bit: bool = False

    # LoRA configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])


# =============================================================================
# Training Stage Configuration
# =============================================================================
class TrainingStage(Enum):
    """Training stages as per PRD Section 4.1"""
    CPT = "continued_pretraining"
    SFT = "supervised_finetuning"
    DPO = "direct_preference_optimization"
    GRPO = "group_relative_policy_optimization"


@dataclass
class CPTConfig:
    """Continued Pretraining configuration (Stage 1)"""
    # Data mixture as per PRD
    general_code_ratio: float = 0.50  # OLMo3 Dolma mix
    solidity_code_ratio: float = 0.30  # Zellic 514K + Etherscan
    audit_reports_ratio: float = 0.10
    math_reasoning_ratio: float = 0.10

    # Training params
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-5
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_steps: int = 50000  # Adjust based on token budget

    # RTX 4090 optimized
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    effective_batch_size: int = 32


@dataclass
class SFTConfig:
    """Supervised Fine-Tuning configuration (Stage 2)"""
    # Data mixture as per PRD
    labeled_vulns_ratio: float = 0.40  # 5K-10K contracts
    synthetic_vulns_ratio: float = 0.25  # HexaCoder
    clean_contracts_ratio: float = 0.20
    general_code_ratio: float = 0.10
    security_docs_ratio: float = 0.05  # SWC, CWE

    # Training params (PRD Section 4.2)
    learning_rate: float = 2e-5
    num_epochs: int = 2
    warmup_ratio: float = 0.0  # PRD says "not needed"
    weight_decay: float = 0.01

    # RTX 4090 optimized
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 16
    effective_batch_size: int = 32  # Target: 3840-7680 effective

    # LoRA
    use_lora: bool = True
    lora_r: int = 16


@dataclass
class DPOConfig:
    """Direct Preference Optimization configuration (Stage 3a)"""
    # Training params (PRD Section 4.2)
    beta: float = 0.1
    learning_rate: float = 5e-7
    num_epochs: int = 1
    warmup_ratio: float = 0.1

    # RTX 4090 optimized
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_length: int = 2048
    max_prompt_length: int = 1024


@dataclass
class GRPOConfig:
    """Group Relative Policy Optimization configuration (Stage 3b)"""
    # Training params (PRD Section 4.2)
    num_generations: int = 4
    learning_rate: float = 1e-6
    num_epochs: int = 1

    # Reward model
    reward_model: str = "slither_validation"

    # RTX 4090 optimized
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 4


# =============================================================================
# Vulnerability Configuration
# =============================================================================
@dataclass
class VulnerabilityConfig:
    """Vulnerability detection configuration as per PRD Section 3.2"""

    # Primary focus SWC types with target F1 scores
    primary_swc_types: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "SWC-107": {"name": "Reentrancy", "target_f1": 0.80, "cwe": "CWE-841"},
        "SWC-101": {"name": "Integer Overflow/Underflow", "target_f1": 0.75, "cwe": "CWE-190"},
        "SWC-115": {"name": "Access Control", "target_f1": 0.60, "cwe": "CWE-284"},
        "SWC-104": {"name": "Unchecked External Calls", "target_f1": 0.70, "cwe": "CWE-252"},
        "SWC-114": {"name": "Front-Running", "target_f1": 0.65, "cwe": "CWE-362"},
    })

    # Full scope: model predicts all 37 SWC types
    total_swc_types: int = 37

    # Output format template (PRD Section 3.1)
    output_template: str = """Vulnerability: {vuln_type}
CWE: {cwe_id}
SWC: {swc_id}
Lines: {line_range}
Severity: {severity}
Explanation: {explanation}
Remediation: {remediation}"""


# =============================================================================
# Evaluation Configuration
# =============================================================================
@dataclass
class EvalConfig:
    """Evaluation configuration as per PRD Section 6"""

    # Target metrics (PRD Section 6.1)
    target_precision: float = 0.35  # >35%
    target_recall: float = 0.70     # >70%
    target_f1: float = 0.45         # >0.45 macro
    max_false_positive_rate: float = 0.65  # <65%

    # GPT-4 baseline for comparison
    gpt4_precision: float = 0.22
    gpt4_recall: float = 0.88
    gpt4_fpr: float = 0.78

    # Evaluation frequency
    eval_steps: int = 1000  # PRD: every 1K steps
    save_steps: int = 1000
    logging_steps: int = 100


# =============================================================================
# Ablation Grid Configuration
# =============================================================================
@dataclass
class AblationGridConfig:
    """Ablation experiment grid as per PRD Section 7"""

    # Variables to sweep (PRD Section 7.1)
    model_sizes: List[str] = field(default_factory=lambda: [
        "135M", "360M", "500M", "1.5B"
    ])

    token_counts: List[str] = field(default_factory=lambda: [
        "500M", "2B", "10B"
    ])

    learning_rates: List[float] = field(default_factory=lambda: [
        1e-5, 5e-5, 1e-4, 5e-4
    ])

    lora_ranks: List[int] = field(default_factory=lambda: [
        8, 16, 32, 64
    ])

    # Constants (PRD Section 7.2)
    sequence_length: int = 2048
    optimizer: str = "adamw"

    # Go/No-Go criteria (PRD Section 7.3)
    checkpoints: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "day_25": {"model": "360M", "min_f1": 0.20, "action": "Review data quality"},
        "day_40": {"model": "1.5B", "min_f1": 0.35, "action": "Pivot strategy"},
        "day_55": {"model": "3B", "min_f1": 0.45, "action": "Additional DPO"},
    })


# =============================================================================
# RTX 4090 Specific Optimizations
# =============================================================================
@dataclass
class RTX4090Config:
    """Hardware-specific optimizations for RTX 4090"""

    # Memory settings
    vram_gb: int = 24
    max_memory_usage: float = 0.95  # Leave 5% headroom

    # Training optimizations
    use_flash_attention: bool = True
    use_gradient_checkpointing: bool = True
    use_mixed_precision: str = "bf16"

    # Batch size recommendations by model size
    batch_size_by_model: Dict[str, int] = field(default_factory=lambda: {
        "135M": 8,
        "360M": 4,
        "500M": 4,
        "1.5B": 2,
        "1.7B": 2,
        "3B": 1,
    })

    # LoRA settings for memory efficiency
    recommended_lora_r: Dict[str, int] = field(default_factory=lambda: {
        "135M": 32,
        "360M": 32,
        "500M": 16,
        "1.5B": 16,
        "1.7B": 8,
        "3B": 8,
    })


# =============================================================================
# Wandb Configuration
# =============================================================================
@dataclass
class WandbConfig:
    """Weights & Biases experiment tracking"""
    project: str = "smart-contract-vuln-detection"
    entity: Optional[str] = None
    tags: List[str] = field(default_factory=lambda: ["ablation", "solidity", "security"])
    log_model: bool = True


# =============================================================================
# Default Configurations
# =============================================================================
DEFAULT_MODEL_CONFIG = ModelConfig(
    name="smollm2-135m",
    model_id=ModelSize.SMOLLM2_135M.value,
)

DEFAULT_SFT_CONFIG = SFTConfig()
DEFAULT_DPO_CONFIG = DPOConfig()
DEFAULT_GRPO_CONFIG = GRPOConfig()
DEFAULT_EVAL_CONFIG = EvalConfig()
DEFAULT_RTX4090_CONFIG = RTX4090Config()
DEFAULT_WANDB_CONFIG = WandbConfig()
