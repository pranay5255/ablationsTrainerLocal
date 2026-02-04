# Smart Contract Vulnerability Detection Model
## Product Requirements Document

**Version:** 3.0  
**Updated:** 2026-02-04  
**Status:** Post-Training Ablation Phase  
**Methodology:** OLMo 3 Post-Training Pipeline

---

## 1. Executive Summary

This project trains a 1B-3B parameter model for smart contract vulnerability detection, with an additional **single-GPU 7B DPO track** for RTX 4090. The goal remains **>35% precision with >70% recall** through an OLMo 3-inspired post-training pipeline. The focus is on ablation studies across three model variants: **THINK** (reasoning), **INSTRUCT** (fast detection), and **RL-ZERO** (baseline).

### What Changed from v2.0

| Aspect | v2.0 | v3.0 |
|--------|------|------|
| Training Paradigm | Continued Pretraining → SFT → DPO → GRPO | **SFT → DPO → RLVR** (no CPT) |
| Model Variants | Single model | **Three ablations**: THINK, INSTRUCT, RL-ZERO |
| DPO Approach | Standard preference pairs | **Delta Learning**: Strong vs weak model pairs |
| RL Algorithm | GRPO | **OlmoRL**: Continuous batching, active sampling, inflight updates |
| Data Status | Collection phase | **Data gathering complete** |

### Project Scope
- **In Scope:** Post-training ablations (SFT, DPO, RLVR), evaluation harness, model comparison
- **Out of Scope:** Base model pretraining (using existing base models)

---

## 2. Key Decisions Summary

| # | Decision | Choice |
|---|----------|--------|
| 1 | Training Paradigm | **Post-training only**: SFT → DPO → RLVR (no continued pretraining) |
| 2 | Model Variants | **Three ablations**: THINK (reasoning), INSTRUCT (fast), RL-ZERO (baseline) |
| 3 | DPO Approach | **Delta Learning**: Pair strong model (Qwen2.5-Coder-3B) with weak model (0.5B) |
| 4 | RL Algorithm | **OlmoRL**: GRPO + continuous batching + active sampling + inflight updates |
| 5 | Verifiers | **Hybrid**: Slither validation + test execution + LM judge for explanations |
| 6 | Response Length | THINK: 32K, INSTRUCT: 8K, RL-ZERO: 16K tokens |
| 7 | Training Infrastructure | Learner: 2-8 nodes, Actors: 7-20 nodes (vLLM), continuous batching critical |
| 8 | Warm-start Strategy | INSTRUCT starts from THINK SFT checkpoint |
| 9 | Task Definition | **Hybrid**: Multi-label classification + Generative explanation |
| 10 | Vulnerability Scope | 5 primary SWC types, predict all 37 |
| 11 | Single-GPU Track | **7B DPO** on RTX 4090 with 4-bit + LoRA, short context |

---

## 2.1 Single-GPU 7B DPO Track (2026-02-04 Addendum)

This addendum describes a pragmatic path to a **7B model on a single RTX 4090** using the existing SFT/DPO pipeline and the `yudai-swe-agent` tool-use format.

**Core constraints**

1. **Memory:** 4-bit quantization + LoRA, gradient checkpointing on.
2. **Context:** 2K-4K max sequence length to stay within 24GB VRAM.
3. **Training focus:** Short SFT for format alignment, **DPO as the primary stage**.

**Recommended base models**

1. Qwen2.5-Coder-7B (preferred for Solidity/code).
2. Any 7B code model that supports 4-bit + LoRA on RTX 4090.

**High-level workflow**

1. Build SFT/DPO datasets from `smart-contract-data/crawlers/output`.
2. Seed DPO pairs with verified good/bad trajectories from `yudai-swe-agent/rl_results`.
3. Train with `train_sft.py` (short) then `train_dpo.py` (primary).

---

## 3. Model Architecture

### 3.1 Task Definition

**Hybrid: Multi-label classification + Generative explanation**

**THINK Model Output Format:**
```
<think>
Analyzing the contract structure...
I see an external call to `msg.sender.call{value: amount}("")` on line 12.
The state variable `balances[msg.sender]` is updated AFTER this external call on line 15.
This creates a reentrancy vulnerability because an attacker can:
1. Call withdraw()
2. Receive the callback in their fallback function
3. Re-enter withdraw() before balance is set to 0
4. Drain the entire contract
</think>

Vulnerability: Reentrancy
SWC: SWC-107
Lines: 12-15
Severity: High
Explanation: External call precedes state update, enabling recursive withdrawal.
Remediation: Move `balances[msg.sender] = 0` before the external call.
```

**INSTRUCT Model Output Format:**
```
Vulnerability: Reentrancy
SWC: SWC-107
Lines: 12-15
Severity: High
Explanation: External call precedes state update, enabling recursive withdrawal.
Remediation: Move `balances[msg.sender] = 0` before the external call.
```

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

**For Ablations:**

| Model | Size | Purpose |
|-------|------|---------|
| SmolLM2-135M | 135M | Quick iteration, data mix validation |
| SmolLM2-360M | 360M | Scaling validation |
| Qwen2.5-Coder-0.5B | 0.5B | Weak model for Delta Learning DPO |
| SmolLM2-1.7B | 1.7B | Pre-scale validation |
| Qwen2.5-Coder-1.5B | 1.5B | Mid-scale ablations |
| Qwen2.5-Coder-7B | 7B | Single-GPU DPO track (4-bit + LoRA) |

**Final Training:**
- **Primary**: Qwen2.5-Coder-3B (strong model for Delta Learning)
- **Backup**: StarCoder2-3B

---

## 4. Three-Model Post-Training Pipeline

### Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     POST-TRAINING PIPELINE (OLMo 3 Style)               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                │
│  │  THINK      │     │  INSTRUCT   │     │  RL-ZERO    │                │
│  │  (Reasoning)│     │  (Fast)     │     │  (Baseline) │                │
│  └──────┬──────┘     └──────┬──────┘     └──────┬──────┘                │
│         │                   │                   │                        │
│    ┌────▼────┐         ┌────▼────┐         ┌────▼────┐                  │
│    │   SFT   │         │   SFT   │         │   Skip  │                  │
│    │ 32K ctx │─────────▶│  8K ctx │         │   SFT   │                  │
│    │ Traces  │ warmstart│ Concise │         │         │                  │
│    └────┬────┘         └────┬────┘         └────┬────┘                  │
│         │                   │                   │                        │
│    ┌────▼────┐         ┌────▼────┐              │                        │
│    │   DPO   │         │   DPO   │              │                        │
│    │ Delta   │         │ Delta   │              │                        │
│    │ Learn   │         │ +Length │              │                        │
│    └────┬────┘         └────┬────┘              │                        │
│         │                   │                   │                        │
│    ┌────▼────┐         ┌────▼────┐         ┌────▼────┐                  │
│    │  RLVR   │         │  RLVR   │         │  RLVR   │                  │
│    │ OlmoRL  │         │ OlmoRL  │         │ OlmoRL  │                  │
│    │ 32K ctx │         │  8K ctx │         │ 16K ctx │                  │
│    └─────────┘         └─────────┘         └─────────┘                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Stage 1: Supervised Fine-Tuning (SFT)

### Data Source Integration (smart-contract-data)

All raw data should come from `smart-contract-data/crawlers/output` to keep a single source of truth.

Expected layout:

```
smart-contract-data/crawlers/output/
├── repos/vulnerability_datasets/   # SmartBugs, SolidiFI, vulndb, etc.
├── repos/audit_repos/              # Audit reports (optional)
└── datasets/kaggle/                # Labeled CSV datasets
```

**How to point the pipeline at this data**

1. Pass `--data-dir ../smart-contract-data/crawlers/output` when running `run_ablations.py`.
2. Or set `data_dir` directly when calling `create_datasets_for_ablation(...)`.

The `preprocessing/DataLoader` already understands this layout and will pick up SmartBugs, Kaggle CSVs, and (optionally) HuggingFace/Zellic when present.

### 5.1 THINK Model SFT

**Purpose:** Generate detailed reasoning traces before vulnerability detection. The model learns to "think out loud" about contract analysis before producing structured output.

**Data Construction (Target: 100K-150K examples):**

| Source | Examples | Processing |
|--------|----------|------------|
| SmartBugs-curated | 143 → ~2.3K | Generate 16 reasoning traces per vulnerability using GPT-4.1/Claude |
| Kaggle SC Vulnerability | 12K → ~12K | Add synthetic reasoning traces for high-confidence labels |
| DeFiHackLabs | 550 → ~8.8K | Convert exploit PoCs to reasoning traces (16 per incident) |
| Audit Reports | 1000+ → ~15K | Rewrite audit findings as reasoning traces |
| Synthetic Pairs | ~50K | HexaCoder + GPT-4.1 generated with full reasoning |
| General Reasoning | ~20K | Math/code reasoning from DAPO-Math, AceCoder (domain transfer) |

**Implementation note (smart-contract-data):**

1. Use `smart-contract-data/crawlers/output/repos/vulnerability_datasets` as the primary labeled source.
2. Use Kaggle CSVs in `output/datasets/kaggle` for high-volume labeled samples.
3. Generate synthetic reasoning traces from the raw contracts and labels, then format via `preprocessing/formatters.py`.

**Reasoning Trace Generation Prompt:**
```
Analyze this Solidity contract for vulnerabilities. Think through your analysis step by step:

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
Remediation: [Fix suggestion]
```

**Data Filtering Pipeline:**

1. **Completeness Filter:** Remove traces missing think tags, SWC ID, or line numbers
2. **Repetition Filter:** Remove traces with >30% repeated phrases (degenerate reasoning)
3. **Slither Validation:** For vulnerability claims, verify with Slither when possible
4. **Length Filter:** Remove traces <500 tokens (too shallow) or >32K tokens (too verbose)
5. **Decontamination:** Remove any examples overlapping with SmartBugs eval set

**Training Configuration:**

```yaml
think_sft:
  max_seq_length: 32768
  learning_rate_sweep: [1e-5, 2e-5, 5e-5, 1e-4, 5e-4]
  batch_size: 3840-7680 (effective)
  epochs: 2
  warmup_ratio: 0.03
  weight_decay: 0.1
  
  checkpoint_selection:
    - Save checkpoints every 500 steps
    - Evaluate on SmartBugs dev set
    - Average top-3 checkpoints by dev F1 (model souping)
```

**Evaluation Metrics:**
- Pass@k (k=1, 8, 32) on SmartBugs holdout
- Average reasoning trace length
- Reasoning quality (LM judge score)

---

### 5.2 INSTRUCT Model SFT

**Purpose:** Direct vulnerability detection without reasoning traces, optimized for speed and tool use.

**Key Difference from THINK:** 
- **Warm-start from THINK SFT checkpoint** (OLMo 3 Table 29 shows +3.3 point average gain)
- Remove reasoning traces from training data
- Add function-calling/tool-use data for Slither/Foundry integration

**Data Construction (Target: 100K-150K examples):**

| Source | Examples | Processing |
|--------|----------|------------|
| THINK SFT Data | ~100K | Strip think blocks, keep only structured output |
| Function-Calling | ~25K | Slither API calls, Foundry integration, MCP-style tool use |
| Concise Examples | ~25K | Max 512 token responses, direct answers |

**Function-Calling Data Format:**
```
User: Analyze this contract for vulnerabilities. Use available tools if needed.

Contract:
{contract_code}
Assistant: I will analyze this contract for potential vulnerabilities.

<tool_call>
{"name": "run_slither", "arguments": {"contract_path": "contract.sol"}}
</tool_call>

Tool Result: {"vulnerabilities": [{"type": "reentrancy", "line": 12}]}

Based on the Slither analysis and my review:

Vulnerability: Reentrancy
SWC: SWC-107
Lines: 12-15
Severity: High
Explanation: External call precedes state update.
Remediation: Apply checks-effects-interactions pattern.
```

**Training Configuration:**

```yaml
instruct_sft:
  max_seq_length: 8192
  learning_rate_sweep: [1e-5, 2e-5, 5e-5, 1e-4]
  batch_size: 3840-7680 (effective)
  epochs: 2
  warmup_ratio: 0.03
  
  # Warm-start from THINK SFT
  init_checkpoint: "think_sft_best_checkpoint"
```

**Evaluation Metrics:**
- Latency (time to first token, total generation time)
- Tool use success rate
- Accuracy on SmartBugs (compare vs THINK)

---

### 5.3 RL-ZERO Model (Skip SFT)

**Purpose:** Establish baseline for RL-only training directly from base model. This helps study the contribution of SFT and DPO to final performance.

**Key Differences:**
- **No SFT stage** - apply RLVR directly to base model
- Simple zero-shot prompt format (no special tokens like think)
- Aggressively filtered data - only hard examples that base model fails

**Prompt Template:**
```
Analyze the following Solidity smart contract for security vulnerabilities.
Identify any issues and provide:
- Vulnerability type
- SWC ID
- Affected lines
- Severity (High/Medium/Low)
- Explanation
- Remediation

Contract:
{contract_code}

Analysis:
```

This variant provides a critical baseline: if RL-ZERO matches THINK/INSTRUCT performance, SFT/DPO may be unnecessary. If it underperforms significantly, we quantify the value of supervised stages.

---

## 6. Stage 2: Preference Optimization (DPO)

### 6.1 Delta Learning Approach (OLMo 3 Methodology)

**Core Insight:** Standard DPO uses chosen/rejected pairs where both come from similar capability models. OLMo 3 shows that **maximizing the capability delta** between chosen and rejected responses yields better results.

**Delta Learning Setup:**
- **Chosen responses:** Generated by strong model (Qwen2.5-Coder-3B or GPT-4.1)
- **Rejected responses:** Generated by weak model (Qwen2.5-Coder-0.5B or SmolLM2-360M)
- **Rationale:** OLMo 3 Table 21 shows continued SFT on strong model responses may hurt performance, but preference tuning on delta yields gains

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DELTA LEARNING DPO                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Prompt: "Analyze this contract for reentrancy vulnerabilities"        │
│                                                                          │
│   ┌─────────────────────────────┐   ┌─────────────────────────────┐     │
│   │  CHOSEN (Strong Model)      │   │  REJECTED (Weak Model)      │     │
│   │  Qwen2.5-Coder-3B          │   │  Qwen2.5-Coder-0.5B         │     │
│   ├─────────────────────────────┤   ├─────────────────────────────┤     │
│   │  <think>                    │   │  The contract looks safe.   │     │
│   │  Line 12 has external call  │   │  I do not see any issues.   │     │
│   │  before state update...     │   │                             │     │
│   │  </think>                   │   │  Vulnerability: None        │     │
│   │                             │   │                             │     │
│   │  Vulnerability: Reentrancy  │   │                             │     │
│   │  SWC: SWC-107              │   │                             │     │
│   │  Lines: 12-15              │   │                             │     │
│   └─────────────────────────────┘   └─────────────────────────────┘     │
│                                                                          │
│   Delta = Capability gap ensures clear learning signal                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.1.1 Single-GPU DPO Pairing (smart-contract-data + rl_results)

For the 7B single-GPU track, build DPO pairs from:

1. Contracts in `smart-contract-data/crawlers/output/repos/vulnerability_datasets` and Kaggle CSVs in `output/datasets/kaggle`.
2. Tool-use trajectories in `yudai-swe-agent/rl_results`.

**Preference scoring (example):**

```
reward = 1.0 * compilation_passed
        + 1.0 * vulnerability_fixed
        - 0.5 * new_vulns_introduced
        - 0.5 * tool_errors
        - 0.5 * format_violations
```

Use the highest-reward trajectory as `chosen` and the lowest as `rejected` for each prompt.

### 6.1.2 Observed Failure Modes in Current RL Results (2026-02-04)

As of 2026-02-04, `rl_results` contains a single episode (`ep_35037`) and is used here as a preliminary signal.

1. Tool errors from incorrect Slither detector names (`reentrancy` vs `reentrancy-eth`).
2. Compile failures before Foundry config was set.
3. Fix removed the core reentrancy but introduced new Slither findings (low-level calls, pragma/solc warnings).

**Training strategies derived from this:**

1. Prefer trajectories that compile before analysis and fixes.
2. Require correct detector selection (`slither --list-detectors` if unsure).
3. Penalize fixes that introduce new findings unrelated to the target vulnerability.
4. Enforce strict output formatting when the agent requires a single bash block.

### 6.2 THINK Model DPO

**Data Construction (Target: 200K preference pairs):**

| Source | Pairs | Processing |
|--------|-------|------------|
| SmartBugs-curated | ~5K | Strong/weak model responses on same contracts |
| Kaggle SC Vulnerability | ~50K | Delta-aware pairs with GPT-judged selection |
| DeFiHackLabs | ~20K | Multi-turn conversations about exploit analysis |
| Synthetic | ~125K | HexaCoder-generated contracts with delta responses |

**Multi-turn Conversation Data:**
```
Turn 1:
User: Analyze this contract for vulnerabilities.
Assistant: [Initial analysis with reentrancy detection]

Turn 2:
User: Can you explain why the checks-effects-interactions pattern would fix this?
Assistant (Chosen): [Detailed explanation of the pattern...]
Assistant (Rejected): [Vague or incorrect explanation...]
```

**Training Configuration:**

```yaml
think_dpo:
  beta: 0.1
  learning_rate_sweep: [5e-7, 1e-6, 5e-6]
  epochs: 1
  dataset_size_sweep: [50K, 100K, 150K, 200K]  # OLMo 3 Figure 23 shows optimal varies
  
  # Early stopping critical - performance peaks then degrades
  early_stopping:
    monitor: smartbugs_dev_f1
    patience: 3
    mode: max
```

**Key Finding (OLMo 3):** DPO yields gains where continued SFT cannot. The preference signal allows learning from comparison that direct imitation cannot capture.

### 6.3 INSTRUCT Model DPO

**Additional Considerations:**
- **Length Control:** Filter pairs where token_count(chosen) - token_count(rejected) > 100 tokens (prevents length exploitation)
- **Multi-turn emphasis:** More interactive debugging scenarios
- **Combine signals:** Delta-learning heuristic + GPT-judged pairs are complementary (OLMo 3 Table 32)

**Data Construction (Target: 200K preference pairs):**

| Source | Pairs | Processing |
|--------|-------|------------|
| THINK DPO Data | ~150K | Apply length filtering (remove >100 token difference) |
| Tool-use Pairs | ~30K | Correct vs incorrect Slither/Foundry usage |
| Concise Responses | ~20K | Prefer shorter, equally accurate responses |

**Training Configuration:**

```yaml
instruct_dpo:
  beta: 0.1
  learning_rate_sweep: [5e-7, 1e-6, 5e-6]
  epochs: 1
  
  # Length control for stability
  length_filter:
    max_chosen_rejected_diff: 100
    
  # Improves RL stability downstream
  response_length_preference: shorter
```

---

## 7. Stage 3: Reinforcement Learning with Verifiable Rewards (RLVR)

### 7.1 OlmoRL Algorithm

**Base Algorithm:** GRPO with improvements from DAPO, Dr GRPO, and OLMo 3 innovations.

**Key Algorithm Modifications:**

| Modification | Description | Impact |
|--------------|-------------|--------|
| Zero-gradient filtering | Remove batches where all samples have identical rewards | Prevents wasted compute |
| Active sampling | Continuously resample to maintain full batch size | Maintains training efficiency |
| Token-level loss | Normalize by total tokens, not sequences | Avoids length bias |
| No KL loss | Allow less-restricted policy updates | Enables larger improvements |
| Clip-higher | Upper bound 1+εhigh, lower bound 1-εlow (εhigh > εlow) | Encourages exploration |
| Truncated IS | Adjust for vLLM vs training engine differences | Correct gradient estimation |
| No std normalization | Dont normalize advantage by standard deviation | Avoids difficulty bias |

**GRPO with OlmoRL Modifications:**

```python
def olmorl_grpo_loss(
    policy_logprobs,      # Log probs from current policy
    ref_logprobs,         # Log probs from reference policy
    rewards,              # Verifier rewards for each completion
    clip_low=0.2,         # Lower clipping bound
    clip_high=0.28,       # Higher clipping bound (clip-higher)
):
    # Compute advantages without std normalization
    advantages = rewards - rewards.mean()  # No / rewards.std()
    
    # Importance sampling ratio
    ratio = torch.exp(policy_logprobs - ref_logprobs)
    
    # Truncated importance sampling
    ratio = torch.clamp(ratio, min=0.01, max=100)  # Prevent extreme ratios
    
    # Asymmetric clipping (clip-higher)
    clipped_ratio = torch.clamp(ratio, 1 - clip_low, 1 + clip_high)
    
    # Policy gradient loss (token-level)
    pg_loss = -torch.min(ratio * advantages, clipped_ratio * advantages)
    
    # No KL penalty term
    # loss = pg_loss + kl_coef * kl_divergence  # REMOVED
    
    return pg_loss.sum() / total_tokens  # Token-level normalization
```

### 7.2 Verifiers for Smart Contract Domain

**Vulnerability Detection Verifier (Binary):**

```python
def vulnerability_detection_reward(
    model_output: str,
    ground_truth: VulnerabilityLabel
) -> float:
    """
    Reward function for vulnerability detection.
    Returns value in [-1, 1] range.
    """
    # Parse model output
    detected_swc = extract_swc(model_output)
    detected_lines = extract_lines(model_output)
    detected_severity = extract_severity(model_output)
    
    rewards = []
    
    # 1. SWC Type Correctness (primary signal)
    if detected_swc == ground_truth.swc_id:
        rewards.append(1.0)
    elif detected_swc in ground_truth.related_swc_ids:
        rewards.append(0.5)  # Partial credit for related vulnerabilities
    else:
        rewards.append(-1.0)
    
    # 2. Line Number Accuracy (IoU-based)
    if detected_lines and ground_truth.lines:
        line_iou = compute_iou(detected_lines, ground_truth.lines)
        rewards.append(line_iou * 2 - 1)  # Scale to [-1, 1]
    
    # 3. Severity Match
    if detected_severity == ground_truth.severity:
        rewards.append(0.5)
    else:
        rewards.append(-0.25)
    
    # 4. Slither Validation (no false positives on clean code)
    if ground_truth.is_clean:
        if not detected_swc or detected_swc == "None":
            rewards.append(0.5)  # Correctly identified clean code
        else:
            rewards.append(-1.0)  # False positive penalty
    
    return sum(rewards) / len(rewards)
```

**Code Execution Verifier (Test Cases):**

```python
def foundry_test_reward(
    model_fix: str,
    original_contract: str,
    test_suite: str
) -> float:
    """
    Reward based on Foundry test execution.
    Tests whether model-suggested fix resolves vulnerability.
    """
    # Apply model suggested fix
    fixed_contract = apply_fix(original_contract, model_fix)
    
    # Write to temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        write_foundry_project(tmpdir, fixed_contract, test_suite)
        
        # Run forge test
        result = subprocess.run(
            ["forge", "test", "--json"],
            cwd=tmpdir,
            capture_output=True
        )
        
        if result.returncode != 0:
            return -1.0  # Fix broke compilation
        
        test_results = json.loads(result.stdout)
        passed = test_results["passed"]
        total = test_results["total"]
        
        # Binary reward: all tests pass = 1, else proportional
        if passed == total:
            return 1.0
        else:
            return (passed / total) * 2 - 1  # Scale to [-1, 1]
```

**Explanation Quality Verifier (LM Judge):**

```python
def explanation_quality_reward(
    model_explanation: str,
    reference_explanation: str,
    judge_model: str = "Qwen3-32B"
) -> float:
    """
    Use LM judge to score explanation quality.
    """
    judge_prompt = f"""
    Rate the quality of this vulnerability explanation on a scale of 0-10.
    
    Consider:
    1. Technical accuracy
    2. Clarity and readability
    3. Completeness (covers root cause, impact, fix)
    4. Specificity (references actual code lines)
    
    Reference explanation (for context):
    {reference_explanation}
    
    Explanation to rate:
    {model_explanation}
    
    Score (0-10):
    """
    
    score = call_judge_model(judge_prompt, model=judge_model)
    return (score / 10) * 2 - 1  # Scale to [-1, 1]
```

### 7.3 Data Construction

**THINK Model RL Data (Target: 105K prompts):**

| Domain | Prompts | Source | Verifier |
|--------|---------|--------|----------|
| Vulnerability Detection | 40K | SmartBugs, Kaggle, DeFiHackLabs | Binary detection + Slither |
| Code Reasoning | 25K | DAPO-Math (adapted), AceCoder | Execution + SymPy |
| Instruction Following | 20K | IFEval-style constraints | Constraint checking |
| General Chat | 20K | Audit report Q&A, explanation requests | LM judge |

**Offline Filtering (OLMo 3 approach):**
```python
def filter_rl_prompts(prompts, model, threshold=0.625):
    """
    Remove prompts where model already achieves >62.5% pass rate.
    These are too easy and waste RL compute.
    """
    filtered = []
    for prompt in prompts:
        # Generate 8 completions
        completions = model.generate(prompt, n=8)
        rewards = [verifier(c) for c in completions]
        pass_rate = sum(r > 0 for r in rewards) / len(rewards)
        
        if pass_rate <= threshold:
            filtered.append(prompt)
    
    return filtered
```

**INSTRUCT Model RL Data (Target: 172K prompts):**

| Domain | Prompts | Source | Notes |
|--------|---------|--------|-------|
| Vulnerability Detection | 50K | Same sources, easier examples | No offline filtering |
| Tool Use | 40K | Slither/Foundry integration | Tool execution verifier |
| General Chat | 50K | More diverse, shorter responses | LM judge |
| Multi-turn | 32K | Interactive debugging sessions | Trajectory reward |

**RL-ZERO Model RL Data (Target: 13.3K prompts):**

| Domain | Prompts | Source | Notes |
|--------|---------|--------|-------|
| Vulnerability Detection | 8K | Aggressively filtered hard examples | Base model fails these |
| Code Reasoning | 3K | DAPO-Math subset | Math-style reasoning |
| Instruction Following | 2.3K | Simple constraints | Basic IF tasks |

### 7.4 Training Configuration

```yaml
think_rl:
  algorithm: olmorl_grpo
  max_response_length: 32768
  training_steps: 750-2300  # Extended training improves performance
  
  # Generation settings
  num_generations_per_prompt: 4
  temperature: 0.7
  top_p: 0.95
  
  # OlmoRL modifications
  zero_gradient_filtering: true
  active_sampling: true
  token_level_loss: true
  no_kl_loss: true
  clip_low: 0.2
  clip_high: 0.28
  truncated_is: true
  no_std_normalization: true

instruct_rl:
  algorithm: olmorl_grpo
  max_response_length: 8192
  training_steps: 500-1000  # Shorter runs, monitor length explosion
  
  # Same OlmoRL modifications
  # ...
  
  # Length monitoring
  length_explosion_threshold: 2.0  # Stop if avg length doubles

rl_zero:
  algorithm: olmorl_grpo
  max_response_length: 16384
  training_steps: 2000+  # Longer runs needed from base model
  
  # Expect slower initial progress
  warmup_steps: 200
  initial_lr: 1e-6  # Lower initial LR for stability
```

### 7.5 OlmoRL Infrastructure

**Architecture:**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        OlmoRL INFRASTRUCTURE                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                      CENTRALIZED LEARNER                         │   │
│   │                      (2-8 nodes, DeepSpeed)                      │   │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │   │
│   │  │   Policy    │  │  Reference  │  │   Critic    │              │   │
│   │  │   Model     │  │   Model     │  │   (opt)     │              │   │
│   │  └─────────────┘  └─────────────┘  └─────────────┘              │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                              │                                           │
│                    Weight Updates (Inflight)                            │
│                              │                                           │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                    DISTRIBUTED ACTORS                            │   │
│   │                    (7-20 nodes, vLLM)                            │   │
│   │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐            │   │
│   │  │ Actor 1 │  │ Actor 2 │  │ Actor 3 │  │ Actor N │            │   │
│   │  │  vLLM   │  │  vLLM   │  │  vLLM   │  │  vLLM   │            │   │
│   │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘            │   │
│   │       │            │            │            │                  │   │
│   │       └────────────┴────────────┴────────────┘                  │   │
│   │                          │                                       │   │
│   │              Continuous Batching                                 │   │
│   │              (backfill finished generations)                     │   │
│   │                          │                                       │   │
│   │              ┌───────────▼───────────┐                          │   │
│   │              │   Active Sampling     │                          │   │
│   │              │   (resample on zero-  │                          │   │
│   │              │    gradient batches)  │                          │   │
│   │              └───────────────────────┘                          │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│   Key Optimizations:                                                    │
│   • Continuous batching: 54% efficiency gain at 32K context            │
│   • Inflight updates: 4x speedup (no generation pause)                 │
│   • Active sampling: Maintains full batch despite filtering            │
│   • Cost: Inference dominates (5-14x more compute than training)       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**Continuous Batching Implementation:**

```python
class ContinuousBatchingManager:
    """
    Backfill finished generations immediately instead of 
    waiting for entire batch to complete.
    """
    def __init__(self, actors, batch_size, max_length):
        self.actors = actors
        self.batch_size = batch_size
        self.max_length = max_length
        self.active_requests = {}
        self.completed_buffer = []
    
    def generate_batch(self, prompts):
        # Initialize all prompts
        for i, prompt in enumerate(prompts):
            actor = self.actors[i % len(self.actors)]
            self.active_requests[i] = actor.start_generation(prompt)
        
        while len(self.completed_buffer) < len(prompts):
            # Check for completed generations
            for req_id, request in list(self.active_requests.items()):
                if request.is_complete():
                    self.completed_buffer.append(request.result)
                    del self.active_requests[req_id]
                    
                    # Backfill with new prompt if available
                    if self.prompt_queue:
                        new_prompt = self.prompt_queue.pop()
                        actor = self.get_free_actor()
                        self.active_requests[req_id] = actor.start_generation(new_prompt)
        
        return self.completed_buffer
```

---

## 8. Ablation Studies

### 8.1 Model Variant Comparison

| Comparison | Hypothesis | Metrics |
|------------|------------|---------|
| THINK vs INSTRUCT | THINK achieves higher accuracy, INSTRUCT faster | F1, latency, tokens/sec |
| THINK vs RL-ZERO | SFT+DPO provide significant gains | F1 delta, sample efficiency |
| INSTRUCT vs RL-ZERO | Warm-start provides benefit | F1 delta, training stability |

### 8.2 Data Ablations

| Ablation | Variables | Expected Insight |
|----------|-----------|------------------|
| Synthetic vs Real | % synthetic data (0%, 25%, 50%, 75%) | Optimal synthetic ratio |
| Reasoning Trace Length | Short/Medium/Long traces | Impact on detection accuracy |
| Delta Magnitude | Strong-weak gap (0.5B, 1B, 2B gap) | Optimal DPO pairing |
| Domain Mix | Security-only vs mixed (math+code+security) | Transfer learning benefit |

### 8.3 Algorithm Ablations

| Ablation | Variants | Expected Insight |
|----------|----------|------------------|
| OlmoRL Components | With/without continuous batching, active sampling | Efficiency impact |
| Verifier Design | Binary vs continuous rewards | Learning signal quality |
| RL Training Length | 500, 1000, 1500, 2000+ steps | Optimal training duration |
| Clip Asymmetry | clip_high = clip_low vs clip_high > clip_low | Exploration impact |

---

## 9. Evaluation Strategy

### 9.1 Metrics

| Metric | Target | GPT-4 Baseline |
|--------|--------|----------------|
| Precision | >35% | 22% |
| Recall | >70% | 88% |
| F1-score | >0.45 macro | N/A |
| False positive rate | <65% | ~78% |

### 9.2 Evaluation Datasets

| Dataset | Size | Purpose |
|---------|------|---------|
| SmartBugs-curated | 143 contracts | Gold standard |
| SolidiFI | 9,369 bugs | Scale testing |
| DeFiHackLabs | 550+ incidents | Real-world validation |

### 9.3 Model-Specific Metrics

**THINK Model:**
- Pass@k (k=1, 8, 32) - generation diversity
- Average reasoning trace length
- Reasoning quality score (LM judge, 0-10)
- Self-consistency (agreement across multiple generations)

**INSTRUCT Model:**
- Latency (time to first token, total generation time)
- Tool use success rate
- Response conciseness (tokens per accurate detection)

**RL-ZERO Model:**
- Training efficiency (steps to target F1)
- Sample efficiency (examples needed vs THINK/INSTRUCT)
- Stability (variance in performance across runs)

---

## 10. Timeline

| Phase | Days | Focus |
|-------|------|-------|
| 1. SFT Data Preparation | 1-10 | Generate reasoning traces, validate with Slither, create THINK/INSTRUCT splits |
| 2. THINK SFT | 11-20 | Train 3 model sizes, LR sweeps, model souping |
| 3. INSTRUCT SFT | 21-25 | Warm-start from THINK, train with concise data |
| 4. RL-ZERO Setup | 26-30 | Filter hard examples, prepare simple prompts |
| 5. DPO (THINK) | 31-35 | Delta learning pairs, LR/size sweeps |
| 6. DPO (INSTRUCT) | 36-40 | Length-controlled pairs, multi-turn data |
| 7. RLVR (THINK) | 41-55 | OlmoRL training, extended runs for best model |
| 8. RLVR (INSTRUCT) | 56-65 | Shorter runs, monitor length explosion |
| 9. RLVR (RL-ZERO) | 66-75 | Longer runs from base model |
| 10. Ablations & Eval | 76-85 | Compare variants, analyze tradeoffs |

---

## 11. Go/No-Go Criteria

| Checkpoint | Criteria | Action if Fail |
|------------|----------|----------------|
| Day 15 | THINK SFT data quality >90% validated | Review trace generation |
| Day 25 | THINK SFT F1 >0.25 on dev set | Review data mix, LR |
| Day 35 | DPO improves over SFT by >2 points | Check delta magnitude |
| Day 55 | THINK RLVR F1 >0.40 | Extend training, check verifiers |
| Day 75 | At least one variant meets targets | Analyze failure modes |

---

## 12. Repository Structure

```
smart-contract-vuln-model/
├── data/
│   ├── raw/                      # Collected data (already complete)
│   │   ├── smartbugs/
│   │   ├── kaggle/
│   │   ├── defihacklabs/
│   │   └── audit_reports/
│   ├── processed/
│   │   ├── think_sft/            # Reasoning trace data
│   │   ├── instruct_sft/         # Concise response data
│   │   ├── think_dpo/            # Delta learning pairs
│   │   ├── instruct_dpo/         # Length-controlled pairs
│   │   └── rl_prompts/           # Filtered RL prompts
│   └── eval/
│       ├── smartbugs_test/       # Held-out evaluation
│       └── defihacklabs_test/
│
├── configs/
│   ├── sft/
│   │   ├── think_sft.yaml
│   │   └── instruct_sft.yaml
│   ├── dpo/
│   │   ├── think_dpo.yaml
│   │   └── instruct_dpo.yaml
│   └── rl/
│       ├── think_rl.yaml
│       ├── instruct_rl.yaml
│       └── rl_zero.yaml
│
├── src/
│   ├── data_generation/
│   │   ├── trace_generator.py    # Generate reasoning traces
│   │   ├── delta_pair_creator.py # Create DPO pairs
│   │   └── rl_prompt_filter.py   # Filter RL prompts
│   ├── verifiers/
│   │   ├── detection_verifier.py # Binary detection reward
│   │   ├── foundry_verifier.py   # Test execution reward
│   │   └── lm_judge.py           # Explanation quality
│   ├── training/
│   │   ├── olmorl_grpo.py        # OlmoRL implementation
│   │   └── continuous_batching.py
│   └── evaluation/
│       ├── smartbugs_eval.py
│       └── metrics.py
│
├── experiments/
│   ├── ablations/
│   │   ├── data_mix/
│   │   ├── delta_magnitude/
│   │   └── algorithm_variants/
│   └── final/
│       ├── think_final/
│       ├── instruct_final/
│       └── rl_zero_final/
│
├── PRD.md                        # This document
├── TASKS.md                      # Implementation tasks
└── README.md
```

---

## 13. Key Resources

### Training Libraries
- [OLMo-core](https://github.com/allenai/OLMo-core) - SFT training (8x faster than Open Instruct)
- [TRL DPOTrainer](https://huggingface.co/docs/trl) - Preference optimization
- [Open Instruct](https://github.com/allenai/open-instruct) - RLVR with OlmoRL modifications
- [vLLM](https://github.com/vllm-project/vllm) - Actor inference with continuous batching

### Evaluation
- [Slither](https://github.com/crytic/slither) - Static analysis validation
- [Foundry](https://book.getfoundry.sh/) - Test case execution
- [OLMES](https://github.com/allenai/olmes) - Evaluation harness (adapt for smart contracts)

### Reference Implementations
- OLMo 3 Think: github.com/allenai/OLMo-core
- Dolci datasets: github.com/allenai/dolma3
- OlmoRL code: Open Instruct repository

### Base Models
- [SmolLM2](https://huggingface.co/collections/HuggingFaceTB/smollm2-6723884218bcda64b34d7db9) - Ablation models
- [Qwen2.5-Coder](https://huggingface.co/Qwen/Qwen2.5-Coder-3B) - Final training

---

## Appendix A: Verifier Implementation Details

### A.1 SWC ID Extraction

```python
import re

def extract_swc(model_output: str) -> str | None:
    """Extract SWC ID from model output."""
    patterns = [
        r"SWC[:\s]*([0-9]{3})",
        r"SWC-([0-9]{3})",
        r"swc[:\s]*([0-9]{3})",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, model_output)
        if match:
            return f"SWC-{match.group(1)}"
    
    return None
```

### A.2 Line Number IoU

```python
def compute_iou(predicted_lines: set, ground_truth_lines: set) -> float:
    """Compute Intersection over Union for line numbers."""
    if not predicted_lines or not ground_truth_lines:
        return 0.0
    
    intersection = len(predicted_lines & ground_truth_lines)
    union = len(predicted_lines | ground_truth_lines)
    
    return intersection / union if union > 0 else 0.0
```

### A.3 Slither Integration

```python
import subprocess
import json

def run_slither(contract_path: str) -> dict:
    """Run Slither analysis on a contract."""
    result = subprocess.run(
        ["slither", contract_path, "--json", "-"],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        return {"error": result.stderr}
    
    return json.loads(result.stdout)
```

---

## Appendix B: OLMo 3 Key Findings Summary

### B.1 SFT Stage
- Long thinking traces (up to 32K tokens) improve reasoning quality
- Model souping (averaging top checkpoints) provides consistent gains
- Warm-starting INSTRUCT from THINK yields +3.3 point average improvement

### B.2 DPO Stage
- Delta Learning outperforms standard preference pairs
- High contrast between chosen/rejected is critical
- DPO yields gains where continued SFT cannot
- Early stopping is critical - performance peaks then degrades

### B.3 RLVR Stage
- Mixing domains prevents over-optimization on single task
- Extended training (2000+ steps) continues to improve performance
- Preference tuning (DPO) provides stronger RL initialization than SFT alone
- Continuous batching provides 54% efficiency gain
- Inflight updates provide 4x speedup
