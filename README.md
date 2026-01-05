# Smart Contract Vulnerability Detection - Ablation Training

Training infrastructure for smart contract vulnerability detection models using ablation experiments on RTX 4090.

## Overview

This repository contains the training code for fine-tuning small language models (135M - 3B parameters) for smart contract vulnerability detection. The training pipeline focuses on CPT, SFT, and DPO, with evaluation on SmartBugs. Data collection and data mixture design are handled externally.

1. **Continued Pretraining (CPT)** - Train on externally prepared CPT corpora (implementation planned)
2. **Supervised Fine-Tuning (SFT)** - Train on labeled vulnerability data
3. **Direct Preference Optimization (DPO)** - Improve detection accuracy with preference pairs

**Note:** GRPO/rollout-centric training is deferred to a future phase.

## Target Metrics

| Metric | Target | GPT-4 Baseline |
|--------|--------|----------------|
| Precision | >35% | 22% |
| Recall | >70% | 88% |
| F1-score | >0.45 | N/A |
| False Positive Rate | <65% | ~78% |

## Hardware Requirements

- **GPU**: NVIDIA RTX 4090 (24GB VRAM)
- **RAM**: 32GB+ recommended
- **Storage**: 100GB+ for models and datasets

## Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install Unsloth (for 2x faster training)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Install Flash Attention (optional but recommended)
pip install flash-attn --no-build-isolation
```

### 2. Data Inputs (External)

Prepare datasets outside this repo and place them in `/data`. The preprocessing helpers here assume the following layout:

```
/data
├── smartbugs/           # SmartBugs-curated dataset
│   └── dataset/
│       └── <vuln_type>/
│           └── *.sol
├── kaggle/              # Kaggle vulnerability dataset
│   └── contracts.jsonl
├── zellic/              # Zellic smart-contract-fiesta
│   └── *.parquet
└── audit_reports/       # Optional, for CPT when added
    └── *.md
```

### 3. Run Training

```bash
# Run a single ablation experiment
python run_ablations.py --config experiments/ablations/smollm2_135m.yaml

# Run all ablations sequentially
python run_ablations.py --all

# Run specific stage only
python run_ablations.py --config experiments/ablations/smollm2_135m.yaml --stage sft
python run_ablations.py --config experiments/ablations/smollm2_135m.yaml --stage dpo

# Dry run (see what would be executed)
python run_ablations.py --config experiments/ablations/smollm2_135m.yaml --dry-run
```

### 4. Evaluate Results

```bash
# Evaluate a trained model
python experiments/evaluation/smartbugs_eval.py \
    --model checkpoints/smollm2-135m/sft \
    --dataset /data/smartbugs
```

DeFiHackLabs evaluation will be added later.

## Project Structure

```
ablationsLocal/
├── experiments/                 # Experiment configurations
│   ├── ablations/              # Ablation YAML configs
│   │   ├── smollm2_135m.yaml   # SmolLM2-135M config
│   │   ├── smollm2_360m.yaml   # SmolLM2-360M config
│   │   ├── qwen_05b.yaml       # Qwen2.5-0.5B config
│   │   ├── baguettotron_321m.yaml
│   │   └── qwen_coder_15b.yaml
│   ├── final/
│   │   └── qwen_coder_3b.yaml  # Final 3B model config
│   └── evaluation/
│       └── smartbugs_eval.py   # Evaluation harness
│
├── training/                    # Training infrastructure
│   ├── configs/
│   │   └── settings.py         # Training configuration
│   └── scripts/
│       ├── train_sft.py        # SFT training script
│       ├── train_dpo.py        # DPO training script
│       └── train_grpo.py       # Planned usage later
│
├── preprocessing/               # Data preprocessing helpers (optional)
│   ├── data_loader.py          # Data loading utilities
│   ├── formatters.py           # Format data for training
│   └── dataset_builder.py      # Build HF datasets
│
├── checkpoints/                 # Model checkpoints (created)
├── logs/                        # Training logs (created)
├── output/                      # Processed data (created)
│
├── run_ablations.py            # Main orchestrator script
├── requirements.txt            # Python dependencies
└── PRD.md                      # Product requirements
```

## Ablation Models

| Model | Size | Purpose | RTX 4090 Batch Size |
|-------|------|---------|---------------------|
| SmolLM2-135M | 135M | LR sweeps, baseline ablations | 8 |
| SmolLM2-360M | 360M | Scaling validation | 4 |
| Qwen2.5-0.5B | 0.5B | Code model transfer validation | 4 |
| Baguettotron-321M | 321M | Alternative architecture | 4 |
| Qwen2.5-Coder-1.5B | 1.5B | Pre-scale validation | 2 |
| Qwen2.5-Coder-3B | 3B | Final model (4-bit) | 1 |

## Training Configuration

### CPT (Stage 0, planned)

Continued pretraining support is planned for this repo. CPT datasets are assumed to be prepared externally.

### SFT (Stage 1)

```yaml
learning_rate: 2e-5
num_epochs: 2
batch_size: 4  # Effective: 32 with grad_accum=8
warmup_ratio: 0.0
weight_decay: 0.01
```

### DPO (Stage 2)

```yaml
beta: 0.1
learning_rate: 5e-7
num_epochs: 1
warmup_ratio: 0.1
```

**Note:** GRPO is deferred; the script exists but is not part of the current pipeline.

## Experiment Tracking

Training metrics are logged to Weights & Biases by default:

```bash
# Login to W&B
wandb login

# Or disable W&B logging
python run_ablations.py --config experiments/ablations/smollm2_135m.yaml --dry-run
```

## Memory Optimization

For larger models on RTX 4090:

1. **4-bit Quantization**: Enable `load_in_4bit: true` in config
2. **Gradient Checkpointing**: Enabled by default
3. **Flash Attention 2**: Automatically used when available
4. **Unsloth**: 2x faster training, 60% less memory

## Go/No-Go Checkpoints

| Checkpoint | Model | Min F1 | Action if Fail |
|------------|-------|--------|----------------|
| Day 25 | 360M | 0.20 | Review data quality |
| Day 40 | 1.5B | 0.35 | Pivot strategy |
| Day 55 | 3B | 0.45 | Additional DPO |

## Output Format

The model outputs vulnerability analysis in this format:

```
Vulnerability: Reentrancy
CWE: CWE-841
SWC: SWC-107
Lines: 12-15
Severity: High
Explanation: The external call on line 12 occurs before state update...
Remediation: Move state update before external call using checks-effects-interactions pattern.
```

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
python run_ablations.py --config experiments/ablations/smollm2_135m.yaml

# Or enable 4-bit quantization in config
load_in_4bit: true
```

### Unsloth Not Available

```bash
# Install Unsloth
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Or training will fall back to standard transformers + PEFT
```

### Missing Data

Ensure data is placed in `/data` directory or specify with `--data-dir`:

```bash
python run_ablations.py --config experiments/ablations/smollm2_135m.yaml --data-dir /path/to/data
```

## License

See individual model licenses:
- SmolLM2: Apache 2.0
- Qwen2.5: Apache 2.0
- Qwen2.5-Coder: Apache 2.0

## References

- [PRD.md](PRD.md) - Full product requirements document
- [TRL Documentation](https://huggingface.co/docs/trl)
- [Unsloth](https://unsloth.ai/)
- [SmartBugs](https://github.com/smartbugs/smartbugs-curated)
