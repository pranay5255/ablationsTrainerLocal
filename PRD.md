# Smart Contract Vulnerability Detection Model
## Product Requirements Document

**Version:** 2.0
**Updated:** 2025-12-29
**Status:** Planning & Data Collection Phase

---

## 1. Executive Summary

This project trains a 1B-3B parameter model for smart contract vulnerability detection, achieving **>35% precision with >70% recall** through a multi-stage training pipeline. The repository handles **data collection** and **experiment planning** only - training execution happens separately.

### Project Scope
- **In Scope:** Data gathering, experiment configuration, ablation planning, evaluation harness setup
- **Out of Scope:** Actual training code execution (separate infrastructure)

---

## 2. Key Decisions Summary

| # | Decision | Choice |
|---|----------|--------|
| 1 | Task Definition | **Hybrid**: Multi-label classification + Generative explanation |
| 2 | Vulnerability Scope | 5 primary SWC types, predict all 37 |
| 3 | Base Models | **Ablations**: SmolLM2-135M/360M, Qwen2.5-0.5B, Baguettotron-321M; **Final**: Qwen2.5-Coder-3B |
| 4 | Training Paradigm | Continued Pretraining → SFT → DPO → GRPO |
| 5 | Data Mixture | OLMo3 Dolma mix + custom Solidity corpus |
| 6 | Evaluation | GRPO-style reward modeling on DeFiHackLabs |
| 7 | Synthetic Data | HexaCoder + mini-swe-agent + Foundry validation |
| 8 | Ablation Grid | Model size × Tokens × Data mix × LR × LoRA rank + eval metrics |
| 9 | Post-training | SFT + DPO + GRPO |
| 10 | Release | Not prioritized for now |

---

## 3. Model Architecture

### 3.1 Task Definition
**Hybrid: Multi-label classification + Generative explanation**

Output format:
```
Vulnerability: Reentrancy
CWE: CWE-841
SWC: SWC-107
Lines: 12-15
Severity: High
Explanation: The external call on line 12 occurs before state update...
Remediation: Move state update before external call using checks-effects-interactions pattern.
```

**Two-step workflow:**
1. **Step 1**: Fine-tuned small model (1B-3B) for detection + explanation
2. **Step 2**: Larger API models (via OpenRouter) for code generation/fixes

### 3.2 Vulnerability Scope

**Primary Focus (evaluation targets):**

| SWC ID | Type | Target F1 | Data Availability |
|--------|------|-----------|-------------------|
| SWC-107 | Reentrancy | >0.80 | Excellent |
| SWC-101 | Integer Overflow/Underflow | >0.75 | Good |
| SWC-115 | Access Control | >0.60 | Medium |
| SWC-104 | Unchecked External Calls | >0.70 | Good |
| SWC-114 | Front-Running | >0.65 | Growing |

**Full scope:** Model will attempt to predict all 37 SWC types, but evaluation focuses on the 5 above.

### 3.3 Base Model Selection

**Ablation Track:**

| Model | Size | Purpose | Tokens |
|-------|------|---------|--------|
| SmolLM2-135M | 135M | Data mix optimization, LR sweeps | 2B |
| SmolLM2-360M | 360M | Scaling validation | 2B |
| Qwen2.5-0.5B | 0.5B | Code model transfer validation | 2B |
| PleIAs/Baguettotron-321M | 321M | Alternative architecture | 2B |
| SmolLM2-1.7B | 1.7B | Pre-scale validation | 10B |
| Qwen2.5-Coder-1.5B | 1.5B | Final pre-scale validation | 10B |

**Final Training:**
- **Primary**: Qwen2.5-Coder-3B (88.4% HumanEval at 7B scale, Apache 2.0)
- **Backup**: StarCoder2-3B (full data tracing, OpenRAIL-M)

---

## 4. Training Pipeline

### 4.1 Three-Stage Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Stage 1: Continued Pretraining (10-50B tokens)                 │
│  ├── 50% General high-quality code (OLMo3 Dolma mix)            │
│  ├── 30% Solidity code (Zellic 514K + Etherscan)                │
│  ├── 10% Audit reports & documentation                          │
│  └── 10% Math/reasoning traces                                   │
│                                                                  │
│  Stage 2: Supervised Fine-Tuning (2 epochs)                     │
│  ├── 40% Vulnerability-labeled Solidity (5K-10K)                │
│  ├── 25% Synthetic vulnerability pairs (HexaCoder)              │
│  ├── 20% Clean non-vulnerable contracts                         │
│  ├── 10% General code instruction data                          │
│  └── 5% Security documentation (SWC, CWE)                        │
│                                                                  │
│  Stage 3: Preference Optimization                                │
│  ├── DPO: 1 epoch with chosen/rejected pairs                    │
│  └── GRPO: Reward modeling on DeFiHackLabs                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Training Libraries

**Recommended Stack:**

| Stage | Library | Rationale |
|-------|---------|-----------|
| Continued Pretraining | **LLaMA-Factory** | Unified interface, CPT support |
| SFT | **Unsloth + TRL** | 2x speed, 60% less memory |
| DPO | **TRL DPOTrainer** | Native support, well-documented |
| GRPO | **TRL GRPOTrainer** | DeepSeek R1 methodology |
| Multi-GPU | **Axolotl** | Best multi-GPU support |

**Alternative: Single unified framework**
- **LLaMA-Factory**: Supports CPT → SFT → DPO in one interface
- Pros: Single config, unified API
- Cons: Less optimization than specialized tools

**Key hyperparameters:**

```yaml
# SFT
sft:
  learning_rate: 2e-5
  batch_size: 3840-7680 (effective)
  epochs: 1-2
  warmup: not needed

# DPO
dpo:
  beta: 0.1
  learning_rate: 5e-7
  epochs: 1

# GRPO
grpo:
  num_generations: 4
  reward_model: slither_validation
```

---

## 5. Data Strategy

### 5.1 Data Sources (from this repo's collection)

| Category | Source | Size | Priority |
|----------|--------|------|----------|
| **Labeled Vulns** | SmartBugs-curated | 143 contracts, 208 vulns | HIGH |
| **Labeled Vulns** | Kaggle SC Vulnerability | 12K+ contracts | HIGH |
| **Large-scale Solidity** | Zellic/smart-contract-fiesta | 514K deduplicated | HIGH |
| **Audit Reports** | Code4rena, Sherlock, Pashov | 1000+ reports | HIGH |
| **Exploits** | DeFiHackLabs | 550+ incidents | HIGH |
| **Educational** | RareSkills, Cyfrin, Ethernaut | Various | MEDIUM |

### 5.2 Synthetic Data Generation

**Enhanced HexaCoder Pipeline with mini-swe-agent:**

```
┌─────────────────────────────────────────────────────────────────┐
│                SYNTHETIC DATA GENERATION                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Generate vulnerable code (GPT-4/Claude via OpenRouter)      │
│     └── Prompt with SWC category + complexity level             │
│                                                                  │
│  2. Validate with static analyzers                               │
│     ├── Slither (primary)                                        │
│     └── Mythril (secondary)                                      │
│                                                                  │
│  3. Generate fixes using mini-swe-agent                          │
│     └── Orchestrate fix generation + validation loop             │
│                                                                  │
│  4. Validate fixes with Foundry                                  │
│     ├── Write forge test scripts                                 │
│     └── Run `forge test` to verify fix                           │
│                                                                  │
│  5. Create training pairs                                        │
│     ├── (vulnerable_code, detection_output) for SFT             │
│     └── (correct_detection, wrong_detection) for DPO            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Target:** 5K-10K validated synthetic pairs

---

## 6. Evaluation Strategy

### 6.1 Metrics

| Metric | Target | GPT-4 Baseline |
|--------|--------|----------------|
| Precision | >35% | 22% |
| Recall | >70% | 88% |
| F1-score | >0.45 macro | N/A |
| False positive rate | <65% | ~78% |

### 6.2 Evaluation Datasets

| Dataset | Size | Purpose |
|---------|------|---------|
| SmartBugs-curated | 143 contracts | Gold standard |
| SolidiFI | 9,369 bugs | Scale testing |
| DeFiHackLabs | 550+ incidents | Real-world validation |

### 6.3 GRPO Reward Modeling

**Using DeFiHackLabs as evaluation environment:**

```python
# Reward function for GRPO
def vulnerability_detection_reward(model_output, ground_truth):
    rewards = []

    # Correctness reward
    if correct_swc_id(model_output, ground_truth):
        rewards.append(1.0)
    else:
        rewards.append(-1.0)

    # Line number accuracy
    line_accuracy = iou(model_output.lines, ground_truth.lines)
    rewards.append(line_accuracy)

    # Slither validation (no false positives on clean code)
    if slither_validates(model_output):
        rewards.append(0.5)

    return sum(rewards) / len(rewards)
```

---

## 7. Ablation Grid

### 7.1 Variables to Sweep

| Variable | Values | Priority |
|----------|--------|----------|
| Model size | 135M, 360M, 500M, 1.5B | HIGH |
| Token count | 500M, 2B, 10B | HIGH |
| Data mix ratios | 5 variants | HIGH |
| Learning rate | 1e-5, 5e-5, 1e-4, 5e-4 | MEDIUM |
| LoRA rank | 8, 16, 32, 64 | MEDIUM |
| **Eval metrics** | SmartBugs F1 per checkpoint | HIGH |

### 7.2 Constants

- Tokenizer: Base model's
- Sequence length: 2048
- Optimizer: AdamW
- Evaluation: SmartBugs-curated (every 1K steps)

### 7.3 Go/No-Go Criteria

| Checkpoint | Criteria | Action if Fail |
|------------|----------|----------------|
| Day 25 | 360M model F1 >0.20 | Review data quality |
| Day 40 | 1.5B model F1 >0.35 | Pivot strategy |
| Day 55 | 3B model F1 >0.45 | Additional DPO |

---

## 8. Repository Structure

```
smart-contract-data/
├── crawlers/                    # Data collection package
│   ├── config/
│   │   ├── settings.py          # Environment, paths, rate limits
│   │   └── sources.yaml         # 40+ data source definitions
│   ├── cloners/                 # GitHub repository handlers
│   ├── scrapers/                # Web scraping (Code4rena, etc.)
│   ├── downloaders/             # Kaggle, HuggingFace
│   ├── processors/              # Normalize to JSONL
│   └── utils/                   # Helpers, logging
│
├── experiments/                 # Experiment configurations (NEW)
│   ├── ablations/
│   │   ├── smollm2_135m.yaml
│   │   ├── smollm2_360m.yaml
│   │   ├── qwen_0.5b.yaml
│   │   └── qwen_coder_1.5b.yaml
│   ├── final/
│   │   └── qwen_coder_3b.yaml
│   └── evaluation/
│       ├── smartbugs_eval.py
│       └── defihacklabs_eval.py
│
├── synthetic/                   # Synthetic data generation (NEW)
│   ├── hexacoder_pipeline.py
│   ├── mini_swe_agent_runner.py
│   └── foundry_validator.py
│
├── output/                      # Collected data
│   ├── repos/
│   ├── reports/
│   ├── datasets/
│   └── processed/
│
├── PRD.md                       # This document
├── TASKS.md                     # Implementation tasks
├── CLAUDE.md                    # AI assistant guidance
└── README.md                    # User documentation
```

---

## 9. Timeline

| Phase | Days | Focus |
|-------|------|-------|
| 1. Data Collection | 1-15 | Clone repos, scrape audits, download datasets |
| 2. Data Processing | 16-25 | Normalize, dedupe, create training format |
| 3. Synthetic Generation | 26-35 | HexaCoder pipeline, validation |
| 4. Experiment Config | 36-40 | Ablation configs, evaluation harness |
| 5. Ablation Execution | 41-55 | Run on RTX 4090 (external) |
| 6. Final Training | 56-70 | Run on 8xH100 (external) |
| 7. Evaluation | 71-75 | SmartBugs, DeFiHackLabs validation |

---

## 10. Key Resources

### Training Libraries
- [TRL (Hugging Face)](https://huggingface.co/docs/trl) - SFT, DPO, GRPO trainers
- [Unsloth](https://unsloth.ai/) - 2x speed, 60% less memory
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) - Unified fine-tuning
- [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) - Multi-GPU support

### Evaluation
- [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) - 100-line agent, >74% SWE-bench
- [SWE-bench](https://www.swebench.com/) - Code evaluation harness
- [Slither](https://github.com/crytic/slither) - Static analyzer validation

### Base Models
- [SmolLM2](https://huggingface.co/collections/HuggingFaceTB/smollm2-6723884218bcda64b34d7db9) - Ablation models
- [Qwen2.5-Coder](https://huggingface.co/Qwen/Qwen2.5-Coder-3B) - Final training
- [OLMo 3](https://allenai.org/olmo) - Methodology reference

### Datasets
- [Zellic/smart-contract-fiesta](https://huggingface.co/datasets/Zellic/smart-contract-fiesta) - 514K contracts
- [SmartBugs-curated](https://github.com/smartbugs/smartbugs-curated) - 143 annotated contracts
- [DeFiHackLabs](https://github.com/SunWeb3Sec/DeFiHackLabs) - 550+ exploit PoCs
