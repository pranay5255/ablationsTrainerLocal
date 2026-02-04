# Smart Contract Vulnerability Detection - Ablation Training

Training infrastructure for smart contract vulnerability detection models using ablation experiments on RTX 4090.

## Overview

This repository contains the training code for fine-tuning small language models (135M - 7B parameters) for smart contract vulnerability detection. The training pipeline focuses on SFT and DPO (with optional CPT), with evaluation on SmartBugs. Data collection and data mixture design are handled externally.

1. **Continued Pretraining (CPT)** - Train on externally prepared CPT corpora (implementation planned)
2. **Supervised Fine-Tuning (SFT)** - Train on labeled vulnerability data
3. **Direct Preference Optimization (DPO)** - Improve detection accuracy with preference pairs

**Note:** GRPO/rollout-centric training is deferred to a future phase. The 7B single-GPU track focuses on DPO with 4-bit + LoRA.

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
python --version  # 3.10 - 3.12 recommended
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install base dependencies
uv pip install -r requirements.txt
# or: pip install -r requirements.txt

# Install GPU/acceleration extras (optional)
uv pip install -r requirements-gpu.txt

# Install Flash Attention (optional but recommended)
uv pip install flash-attn --no-build-isolation
```

### 2. Data Inputs (smart-contract-data)

This repo expects the raw data to live in the same layout produced by `smart-contract-data/crawlers/output`. Point `--data-dir` to that folder.

```
smart-contract-data/crawlers/output/
├── repos/
│   ├── vulnerability_datasets/   # smartbugs-curated, SolidiFI, vulndb, etc.
│   └── audit_repos/               # audit reports (optional)
└── datasets/
    ├── kaggle/                    # labeled CSV datasets
    └── huggingface/               # Zellic/smart-contract-fiesta (optional, large)
```

### 3. Run Training

```bash
# Run a single ablation experiment
uv run python run_ablations.py --config experiments/ablations/smollm2_135m.yaml

# Run all ablations sequentially
uv run python run_ablations.py --all

# Run specific stage only
uv run python run_ablations.py --config experiments/ablations/smollm2_135m.yaml --stage sft
uv run python run_ablations.py --config experiments/ablations/smollm2_135m.yaml --stage dpo

# Dry run (see what would be executed)
uv run python run_ablations.py --config experiments/ablations/smollm2_135m.yaml --dry-run
```

### 4. Evaluate Results

```bash
# Evaluate a trained model
uv run python experiments/evaluation/smartbugs_eval.py \
    --model checkpoints/smollm2-135m/sft \
    --dataset /path/to/smart-contract-data/crawlers/output/repos/vulnerability_datasets/smartbugs-curated
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
| Qwen2.5-Coder-7B | 7B | Single-GPU DPO (4-bit + LoRA) | 1 |

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
uv run python run_ablations.py --config experiments/ablations/smollm2_135m.yaml --dry-run
```

## Memory Optimization

For larger models on RTX 4090 (including 7B):

1. **4-bit Quantization**: Enable `load_in_4bit: true` in config
2. **Gradient Checkpointing**: Enabled by default
3. **Flash Attention 2**: Automatically used when available
4. **Unsloth**: 2x faster training, 60% less memory

## 7B Single-GPU Track (DPO-Focused)

Recommended when targeting a 7B model on one RTX 4090.

1. Use 4-bit + LoRA and keep max sequence length modest (2K-4K).
2. Prefer DPO over long SFT runs; use SFT only for format alignment.
3. Keep per-device batch size at 1 and increase gradient accumulation.

Example config deltas:

```yaml
model:
  name: "Qwen/Qwen2.5-Coder-7B"
  load_in_4bit: true
  max_seq_length: 2048
lora:
  enabled: true
  r: 16
  alpha: 32
stages:
  sft:
    num_epochs: 1
    per_device_batch_size: 1
    gradient_accumulation_steps: 32
  dpo:
    per_device_batch_size: 1
    gradient_accumulation_steps: 32
```

## Using smart-contract-data

Point `--data-dir` to the crawler output directory:

```bash
uv run python run_ablations.py \
  --config experiments/ablations/smollm2_135m.yaml \
  --data-dir ../smart-contract-data/crawlers/output
```

Synthetic data plan (SFT/DPO):

1. Pull contracts from `repos/vulnerability_datasets` and Kaggle CSVs in `datasets/kaggle`.
2. Use `preprocessing/dataset_builder.py` to format SFT/DPO examples.
3. For DPO, create preference pairs with a verifier:
4. Chosen = compiles, fixes vuln, no new findings.
5. Rejected = compile failures, new findings, or wrong vulnerability.
6. Use `yudai-swe-agent/rl_results` to seed bad/good pairs and tool-use traces.

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
uv run python run_ablations.py --config experiments/ablations/smollm2_135m.yaml

# Or enable 4-bit quantization in config
load_in_4bit: true
```

### Unsloth Not Available

```bash
# Install Unsloth
uv pip install -r requirements-gpu.txt

# Or training will fall back to standard transformers + PEFT
```

### Missing Data

Ensure data is placed in `/data` directory or specify with `--data-dir`:

```bash
uv run python run_ablations.py --config experiments/ablations/smollm2_135m.yaml --data-dir /path/to/data
```

## License

See individual model licenses:
- SmolLM2: Apache 2.0
- Qwen2.5: Apache 2.0
- Qwen2.5-Coder: Apache 2.0

## References

- [PRD_v3.md](PRD_v3.md) - Full product requirements document
- [TRL Documentation](https://huggingface.co/docs/trl)
- [Unsloth](https://unsloth.ai/)
- [SmartBugs](https://github.com/smartbugs/smartbugs-curated)
