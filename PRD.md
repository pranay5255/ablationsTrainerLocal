# Smart Contract Vulnerability Detection Model
## Product Requirements Document

**Version:** 2.0
**Updated:** 2025-12-29
**Status:** Training Infrastructure Phase

---

## 1. Executive Summary

This project trains a 1B-3B parameter model for smart contract vulnerability detection, achieving **>35% precision with >70% recall** through a multi-stage training pipeline (CPT → SFT → DPO). The repository provides **training code**, **experiment configs**, and **evaluation harnesses**; data collection and data mixture design are handled externally.

### Project Scope
- **In Scope:** Training code for CPT/SFT/DPO, experiment configuration, ablation planning, evaluation harness setup
- **Out of Scope:** Data collection/crawlers, synthetic data generation, rollout-centric training (GRPO) for now, release/deployment

---

## 2. Key Decisions Summary

| # | Decision | Choice |
|---|----------|--------|
| 1 | Base Models | **Ablations**: SmolLM2-135M/360M, Qwen2.5-0.5B, Baguettotron-321M; **Final**: Qwen2.5-Coder-3B |
| 2 | Training Paradigm | Continued Pretraining → SFT → DPO (GRPO later) |
| 3 | Evaluation | SmartBugs now; DeFiHackLabs later |
| 4 | Ablation Grid | Model size × Tokens × LR × LoRA rank + eval metrics |
| 5 | Post-training | SFT + DPO (GRPO later) |
| 6 | Release | Not prioritized for now |

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

### 4.1 Three-Stage Pipeline (Current Focus)

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Stage 1: Continued Pretraining (10-50B tokens)                 │
│  └── Uses externally prepared CPT dataset (mixture defined out  │
│      of repo scope)                                             │
│                                                                  │
│  Stage 2: Supervised Fine-Tuning (2 epochs)                     │
│  └── Uses externally prepared labeled dataset                   │
│                                                                  │
│  Stage 3: Preference Optimization                                │
│  └── DPO: 1 epoch with chosen/rejected pairs                    │
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

```

**Note:** GRPO/rollout-centric training is planned for a future phase once CPT/SFT/DPO baselines are validated.

---

## 5. Data Inputs (External)

Training and evaluation datasets are prepared outside this repository. This repo assumes datasets are already available on disk (see `README.md` for the expected layout). Data collection, crawling, and synthetic generation are out of scope here.

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
| DeFiHackLabs | 550+ incidents | Real-world validation (planned) |

---

## 7. Ablation Grid

### 7.1 Variables to Sweep

| Variable | Values | Priority |
|----------|--------|----------|
| Model size | 135M, 360M, 500M, 1.5B | HIGH |
| Token count | 500M, 2B, 10B | HIGH |
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
ablationsLocal/
├── experiments/                 # Experiment configurations
│   ├── ablations/
│   │   ├── smollm2_135m.yaml
│   │   ├── smollm2_360m.yaml
│   │   ├── qwen_05b.yaml
│   │   └── qwen_coder_15b.yaml
│   ├── final/
│   │   └── qwen_coder_3b.yaml
│   └── evaluation/
│       └── smartbugs_eval.py
│
├── training/                    # Training infrastructure
│   ├── configs/
│   │   └── settings.py
│   └── scripts/
│       ├── train_sft.py
│       ├── train_dpo.py
│       └── train_grpo.py         # Planned usage later
│
├── preprocessing/               # Data prep helpers (expects external data)
│   ├── data_loader.py
│   ├── dataset_builder.py
│   └── formatters.py
│
├── checkpoints/                 # Model checkpoints (created)
├── logs/                        # Training logs (created)
├── run_ablations.py             # Orchestrator
├── PRD.md                       # This document
└── README.md                    # User documentation
```

---

## 9. Timeline

| Phase | Days | Focus |
|-------|------|-------|
| 1. CPT Setup | 1-15 | Continued pretraining config + baseline run |
| 2. SFT Ablations | 16-35 | SFT training across ablation configs |
| 3. DPO Tuning | 36-45 | Preference optimization on chosen runs |
| 4. Evaluation | 46-55 | SmartBugs validation + metric review |
| 5. DeFiHackLabs Eval (Later) | TBD | Add harness and run real-world validation |

---

## 10. Key Resources

### Training Libraries
- [TRL (Hugging Face)](https://huggingface.co/docs/trl) - SFT and DPO trainers
- [Unsloth](https://unsloth.ai/) - 2x speed, 60% less memory
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) - Unified fine-tuning
- [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) - Multi-GPU support

### Evaluation
- [Slither](https://github.com/crytic/slither) - Static analyzer validation

### Base Models
- [SmolLM2](https://huggingface.co/collections/HuggingFaceTB/smollm2-6723884218bcda64b34d7db9) - Ablation models
- [Qwen2.5-Coder](https://huggingface.co/Qwen/Qwen2.5-Coder-3B) - Final training
- [OLMo 3](https://allenai.org/olmo) - Methodology reference
