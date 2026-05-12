# Speculative Decoding

A from-scratch speculative decoding inference engine built in PyTorch.  
The goal is to deeply understand, implement, and benchmark speculative decoding — measuring real latency, token acceptance rates, and speedup across multiple draft/target model pairs.

> **Status:** Infrastructure, data pipeline, and model layer complete. Core decoding engine in progress.

---

## Table of Contents

- [What is Speculative Decoding?](#what-is-speculative-decoding)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [What You Can Run Right Now](#what-you-can-run-right-now)
- [Model Pairs](#model-pairs)
- [Roadmap](#roadmap)
- [Theory Reference](#theory-reference)

---

## What is Speculative Decoding?

Autoregressive LLM inference is memory-bandwidth-bound at batch size 1: generating each token requires streaming the entire model's weights through GPU memory, leaving compute idle ~99% of the time.

Speculative decoding exploits this by running a cheap **draft model** to propose K candidate tokens, then verifying all K with the **target model** in a single parallel forward pass — the same memory cost as generating one token. A rejection sampling scheme guarantees the output distribution is identical to running the target model alone.

At a 70–80% token acceptance rate with a 10–20× cheaper draft model, this yields **2–3× wall-clock speedup** with no change in output quality.

For the full derivation with worked numerical examples, see [`docs/speculative_decoding_theory.md`](docs/speculative_decoding_theory.md).

---

## Project Structure

```
speculative-decoding/
│
├── src/
│   ├── models/
│   │   ├── registry.py        ✅  ModelPair dataclass + 4 registered pairs
│   │   ├── loader.py          ✅  load_model() — HuggingFace weights + dtype handling
│   │   ├── ollama_loader.py   ✅  Ollama convenience wrapper (hardware validation only)
│   │   └── wrapper.py         ✅  ModelWrapper — forward pass + KV cache management
│   │
│   ├── decoding/
│   │   ├── autoregressive.py  🔲  Baseline token-by-token generation
│   │   ├── speculative.py     🔲  Speculative decoding loop
│   │   └── rejection.py       🔲  Rejection sampling + adjusted distribution
│   │
│   ├── profiling/
│   │   ├── timer.py           🔲  CUDA-aware latency timers
│   │   ├── memory.py          🔲  GPU memory tracking
│   │   └── metrics.py         🔲  Token acceptance rate, speedup aggregation
│   │
│   ├── data/
│   │   └── prompts.py         ✅  PromptDataset — 150 prompts across 3 domains
│   │
│   └── utils/
│       ├── logging.py         🔲  Structured logging
│       └── reproducibility.py 🔲  Seed control, deterministic flags
│
├── configs/                   🔲  YAML experiment configs (not yet populated)
├── benchmarks/                🔲  Benchmark entry-point scripts
├── tests/
│   └── test_kv_cache.py       ✅  17 integration tests for KV cache shape + truncation
├── notebooks/                 🔲  Analysis notebooks
├── results/                   (gitignored — generated outputs)
├── figures/                   (gitignored — generated plots)
│
├── docs/
│   └── speculative_decoding_theory.md  ✅  Theory deep-dive with worked examples
│
├── environment.yml            ✅  Conda env (Python 3.11, PyTorch 2.3, CUDA 12.1)
└── pyproject.toml             ✅  Package metadata + src layout
```

`✅` = implemented and runnable  `🔲` = scaffolded, implementation pending

---

## Setup

### Prerequisites

- [Conda](https://docs.conda.io/en/latest/miniconda.html) (or Mamba)
- CUDA 12.1-compatible GPU recommended; CPU works for `gpt2`-scale models

### 1. Create the environment

```bash
conda env create -f environment.yml
conda activate speculative-decoding
```

### 2. Install the package in editable mode

```bash
pip install -e .
```

### 3. (Optional) Verify PyTorch sees your GPU

```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### 4. (Optional) HuggingFace token for gated models

The `tinyllama_llama3`, `phi3_llama3`, and `llama3_self` pairs require access to
`meta-llama/Meta-Llama-3-8B-Instruct`. Request access at
[huggingface.co/meta-llama](https://huggingface.co/meta-llama), then:

```bash
huggingface-cli login
```

The `gpt2_dev` pair has no access restrictions and works immediately.

---

## What You Can Run Right Now

### Load a model and verify inference

Loads a HuggingFace causal LM, prints parameter count and memory footprint,
then generates a short test sequence to confirm the model is functional.

```bash
# Smallest pair — no token required, fits on any GPU with ~3 GB VRAM
python -m src.models.loader --model gpt2 --dtype float16
python -m src.models.loader --model gpt2-xl --dtype float16

# Larger models (requires HF token + sufficient VRAM)
python -m src.models.loader --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --dtype float16
python -m src.models.loader --model meta-llama/Meta-Llama-3-8B-Instruct --dtype float16
python -m src.models.loader --model meta-llama/Meta-Llama-3-8B-Instruct --dtype int4

# Control how many tokens to generate in the smoke test
python -m src.models.loader --model gpt2 --dtype float16 --n-tokens 20
```

**Expected output:**
```
[loader] tokenizer  ← gpt2
[loader] model      ← gpt2  (dtype=float16)
  model      : gpt2
  parameters : 124.4 M
  memory     : 0.23 GB

[loader] generating 10 tokens …
[loader] output: 'The quick brown fox jumps over the lazy dog ...'
[loader] OK
```

---

### Inspect the model pair registry

```python
from src.models.registry import list_pairs, get_pair

print(list_pairs())
# ['gpt2_dev', 'tinyllama_llama3', 'phi3_llama3', 'llama3_self']

pair = get_pair("gpt2_dev")
print(pair.draft_model_id)   # gpt2
print(pair.target_model_id)  # gpt2-xl
print(pair.draft_dtype)      # float16
```

---

### Build the benchmark prompt dataset

Downloads 150 prompts (50 per domain) from HuggingFace datasets and saves them
as JSON with pre-computed token counts. Falls back to built-in prompts
automatically if a dataset is unavailable.

```bash
# Full dataset (150 prompts) + tiny dev set (15 prompts)
python -m src.data.prompts --tokenizer gpt2 --output data/prompts.json

# Use a different tokenizer (important: token counts depend on the tokenizer)
python -m src.data.prompts --tokenizer TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

**Expected output:**
```
[prompts] loading tokenizer: gpt2
[prompts] building full dataset (50 prompts per domain) …

domain                  n    min    mean    max
-----------------------------------------------
code                   50     34    89.4    213
conversation           50     10    24.7     76
summarization          50    132   158.2    187

[prompts] saved 150 prompts → data/prompts.json
[prompts] saved  15 prompts → data/prompts_tiny.json
```

**Load saved prompts in your own code:**
```python
from src.data.prompts import PromptDataset

dataset = PromptDataset.load("data/prompts.json")

code_prompts = dataset.get_by_domain("code")       # list[Prompt]
all_prompts  = dataset.get_all()                    # list[Prompt]

print(len(dataset))                                 # 150
print(code_prompts[0].prompt_id)                    # "code_000"
print(code_prompts[0].token_count)
```

**Prompt domains:**

| Domain | Source | Format |
|---|---|---|
| `code` | `openai/openai_humaneval` | Python function signature + docstring |
| `conversation` | `tatsu-lab/alpaca` | Instruction text |
| `summarization` | `cnn_dailymail 3.0.0` | Article (≤512 tokens) + "Summarize the above article:" |

---

### Use ModelWrapper for forward passes and KV cache control

`ModelWrapper` is the interface every decoding component uses to talk to a
model. It exposes forward passes, cache length queries, and — critically —
`truncate_cache()`, which rolls the KV cache back to an arbitrary position
when speculative decoding rejects a draft token.

```python
import torch
from src.models.loader import load_model
from src.models.wrapper import ModelWrapper

model, tokenizer = load_model("gpt2", "float32")
device = next(model.parameters()).device
wrapper = ModelWrapper(model, tokenizer, device)

# --- Prefill ---
prompt_ids = tokenizer.encode("The quick brown fox", return_tensors="pt").to(device)
logits, cache = wrapper.forward(prompt_ids)

print(wrapper.get_cache_length(cache))   # 4  (one entry per prompt token)
# logits shape: [1, 4, 50257]

# --- Decode one token ---
next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)   # greedy
logits, cache = wrapper.forward(next_token, past_key_values=cache)
print(wrapper.get_cache_length(cache))   # 5

# --- Simulate rejection at position 3: roll back to position 3 ---
cache = wrapper.truncate_cache(cache, keep_length=3)
print(wrapper.get_cache_length(cache))   # 3

# Continue from position 3 with a replacement token
replacement = torch.tensor([[1234]]).to(device)
logits, cache = wrapper.forward(replacement, past_key_values=cache)
print(wrapper.get_cache_length(cache))   # 4
```

**KV cache shape** — every tensor inside `past_key_values` has shape
`[batch, num_heads, seq_len, head_dim]`. For GPT-2 small that is
`[1, 12, seq_len, 64]` per layer across 12 layers.

---

### Run the test suite

```bash
# All tests (requires GPT-2 weights — ~500 MB download on first run)
pytest tests/test_kv_cache.py -v

# Just shape checks (faster)
pytest tests/test_kv_cache.py -v -k "Shape"
```

**17 tests across 4 classes:**

| Class | What it checks |
|---|---|
| `TestCacheShape` | `len(past_kv)==12`, each entry is `(key, value)`, shapes `[1, 12, 20, 64]`, `get_cache_length` |
| `TestCacheTruncation` | Shapes after `keep_length=10`, truncate to 1, original cache unmodified |
| `TestForwardWithCache` | Logits `[1, 1, 50257]` after truncated-cache pass, cache grows by 1, `use_cache=False`→`None` |
| `TestProperties` | `vocab_size`, `param_count`, `memory_footprint_mb`, `repr` |

---

### Quick hardware validation with Ollama

If you have [Ollama](https://ollama.com) installed, use this to check whether
a model fits on your hardware **before** downloading the full HuggingFace
weights. This is a convenience tool only — see
[`src/models/ollama_loader.py`](src/models/ollama_loader.py) for why it cannot
be used in the benchmarking pipeline.

```bash
# Check if Ollama is running and list available models
python -m src.models.ollama_loader

# Test generation on a specific model
python -m src.models.ollama_loader --model llama3:8b --prompt "Explain attention in transformers"

# Adjust generation length
python -m src.models.ollama_loader --model llama3:8b --max-tokens 100
```

**Expected output (status check):**
```
[ollama] checking server at http://localhost:11434 …
[ollama] ✓ server is running.

model                               params   quant  size (GB)
--------------------------------------------------------------
llama3:8b                              8B   Q4_0        4.7
phi3:mini                            3.8B   Q4_0        2.2
```

---

## Model Pairs

Four pairs are registered in [`src/models/registry.py`](src/models/registry.py):

| Pair name | Draft model | Target model | Draft dtype | Notes |
|---|---|---|---|---|
| `gpt2_dev` | `gpt2` (124M) | `gpt2-xl` (1.5B) | fp16 | No token required. Fast to iterate on. |
| `tinyllama_llama3` | `TinyLlama-1.1B-Chat` | `Llama-3-8B-Instruct` | fp16 | Shared BPE vocab → decent acceptance rate |
| `phi3_llama3` | `Phi-3-mini-4k` (3.8B) | `Llama-3-8B-Instruct` | fp16 | Higher draft cost, higher expected acceptance |
| `llama3_self` | `Llama-3-8B-Instruct` (int4) | `Llama-3-8B-Instruct` | int4 / fp16 | Self-speculation: near-perfect acceptance, speedup from quantised draft bandwidth |

VRAM requirements (approximate):

| Pair | Draft VRAM | Target VRAM | Total |
|---|---|---|---|
| `gpt2_dev` | 0.25 GB | 3.0 GB | ~3.3 GB |
| `tinyllama_llama3` | 2.2 GB | 16 GB | ~18 GB |
| `phi3_llama3` | 7.6 GB | 16 GB | ~24 GB |
| `llama3_self` | 4.7 GB (int4) | 16 GB | ~21 GB |

---

## Roadmap

### Phase 1 — Infrastructure ✅
- [x] Project scaffold, environment, packaging
- [x] Model registry with 4 draft/target pairs
- [x] HuggingFace model loader with dtype handling (fp16, bf16, fp32, int4)
- [x] Ollama convenience loader for hardware validation
- [x] 150-prompt benchmark dataset across code / conversation / summarization
- [x] `ModelWrapper` — forward pass, KV cache truncation, cache length utilities
- [x] `tests/test_kv_cache.py` — 17 integration tests covering cache shape, truncation, and forward-with-cache

### Phase 2 — Core Decoding Engine 🔲
- [ ] `src/decoding/autoregressive.py` — baseline greedy/sampling loop with KV cache
- [ ] `src/decoding/rejection.py` — rejection sampling + adjusted distribution
- [ ] `src/decoding/speculative.py` — full speculative decoding loop (K draft tokens → parallel target verification)

### Phase 3 — Profiling 🔲
- [ ] `src/profiling/timer.py` — CUDA event-based latency measurement
- [ ] `src/profiling/memory.py` — peak VRAM tracking per phase
- [ ] `src/profiling/metrics.py` — token acceptance rate, tokens/second, speedup vs baseline

### Phase 4 — Benchmarks & Analysis 🔲
- [ ] End-to-end benchmark runner across all 4 model pairs × 3 domains × K ∈ {1,2,4,8}
- [ ] Acceptance rate vs prompt domain analysis
- [ ] Speedup breakdown: time in draft vs target vs overhead
- [ ] Figures: speedup heatmap, acceptance rate distributions, latency CDFs

---

## Theory Reference

[`docs/speculative_decoding_theory.md`](docs/speculative_decoding_theory.md) covers:

1. **Why autoregressive inference is slow** — arithmetic intensity analysis with concrete numbers for a 7B FP16 model (1 FLOP/byte vs 333 FLOP/byte ridge point, 0.3% GPU compute utilisation)
2. **How speculative decoding works** — complete K=4 walkthrough with a 5-word vocabulary, including rejection decisions with actual probability values
3. **Why rejection sampling preserves the target distribution** — full proof that P(output=x) = p_target(x) in both the p≥q and p<q cases
4. **What determines the speedup** — derivation of `S = (avg_accepted + 1) / (1 + C_draft/T_target)`, numerical examples, and conditions under which speculative decoding hurts

---

## License

MIT — see [LICENSE](LICENSE).
