# Speculative Decoding — Theory and Intuition

A deep-dive into the mechanics of speculative decoding: why autoregressive inference is slow,
how speculative decoding fixes it, why it is theoretically sound, and what determines the speedup.

---

## Table of Contents

1. [Why is autoregressive inference slow?](#1-why-is-autoregressive-inference-slow)
2. [How does speculative decoding work?](#2-how-does-speculative-decoding-work)
3. [Why does rejection sampling preserve the target distribution?](#3-why-does-rejection-sampling-preserve-the-target-distribution)
4. [What determines the speedup?](#4-what-determines-the-speedup)

---

## 1. Why Is Autoregressive Inference Slow?

### The autoregressive loop

At every decode step the model takes the full token sequence, runs a forward pass, and produces
a probability distribution over the vocabulary. You sample one token, append it, and repeat.
There is no way to parallelise across tokens — each token depends on all previous ones.

The key optimisation already in use is the **KV cache**: instead of recomputing keys and values
for all past tokens on every step, you store them and only compute Q/K/V for the single new
token. So at decode step `t`, the GPU receives a **single-token** input vector of shape
`[1, d_model]`.

### What the GPU actually does each step

Take a 7B parameter LLaMA-style model (32 layers, `d_model=4096`, FFN intermediate=11008,
vocab=32000), weights in FP16 → **14 GB total**.

For each of the 32 layers, the GPU must:

1. **Attention projections** — multiply the `[1, 4096]` input against Q, K, V, O weight matrices
   (`[4096, 4096]` each): 4 matrix-vector multiplies.
2. **Attend over the KV cache** — dot the query against all `L` cached keys, weighted-sum the
   values.
3. **FFN** — three matrix-vector multiplies against `gate_proj`, `up_proj`, `down_proj`.

The dominant cost is steps 1 and 3: **GEMV** (matrix-vector multiply) against the full weight
matrices.

### The arithmetic intensity argument

**Arithmetic intensity** = FLOPs ÷ bytes loaded from memory.

| Quantity | Value |
|---|---|
| FLOPs per token (≈ 2N rule) | 2 × 7×10⁹ = **14 GFLOPs** |
| Bytes loaded (all weights, FP16) | 7×10⁹ × 2 = **14 GB** |
| **Arithmetic intensity** | 14 GFLOPs / 14 GB = **1 FLOP/byte** |

The GPU's **ridge point** — the compute-to-bandwidth ratio at which work transitions from
memory-bound to compute-bound:

```
Ridge point = peak compute / peak bandwidth
            = 300 TFLOP/s  /  900 GB/s
            = 333 FLOP/byte
```

The workload has intensity **1 FLOP/byte**. The GPU requires **333 FLOP/byte** to be
compute-bound. The operation sits **333× below the ridge point** — deeply memory-bandwidth-bound.

### Concrete timing and utilisation

**Memory-bound time** (the actual bottleneck):

```
t_memory = 14 GB / 900 GB/s = 15.6 ms per token   (~64 tok/s)
```

**Time if the workload were compute-bound:**

```
t_compute = 14 GFLOPs / 300,000 GFLOP/s = 0.047 ms per token
```

**GPU compute utilisation:**

```
utilisation = t_compute / t_actual = 0.047 ms / 15.6 ms = 0.3%
```

The GPU is arithmetically active **0.3% of the time**. The remaining 99.7% it stalls waiting
for data from HBM.

### Why batch_size=1 is the worst case

At batch_size=1 the input to each weight matrix is a single vector `[1, 4096]` — a **GEMV**.
You load 4096×4096 = 16.7 M numbers to perform 16.7 M multiplications: exactly 1 FLOP/byte.

At batch_size=B the input becomes `[B, 4096]` — a **GEMM**. You still load the same weights
but now perform B×16.7 M FLOPs against them. Arithmetic intensity scales as `B FLOP/byte`.
Once B ≈ 333 on this GPU the model becomes compute-bound and the hardware is fully utilised.

**The fundamental problem**: at batch_size=1 every decode step forces you to stream the entire
14 GB model through memory to do a trivial amount of arithmetic.

---

## 2. How Does Speculative Decoding Work?

### The key insight

The GPU is idle 99.7% of the time, yet each decode step still costs ~15.6 ms to load the
weights. What if you could get **multiple tokens out of a single weight-loading pass**?

Whether you process 1 token or K tokens in a single causal forward pass, you still load all
model weights once. So processing K tokens costs roughly the same as processing 1 token (for
small K where arithmetic intensity stays below the ridge point). If the K tokens turn out to be
correct, you have produced K+1 tokens for the cost of approximately one target-model pass.

The challenge: you only know what those K tokens should be after you have already generated them.  
**Solution**: use a cheap draft model to guess K tokens first, then verify with the target in one shot.

### The algorithm

```
1. Draft model autoregressively generates K candidate tokens x̃₁ … x̃ₖ
2. Target model processes [context, x̃₁, …, x̃ₖ] in one forward pass
3. At each position, use rejection sampling to accept or reject each x̃ᵢ
4. Output the accepted tokens + one bonus/replacement token
```

### Complete example with K=4

**Vocabulary**: `{the:0, cat:1, sat:2, on:3, mat:4}`  
**Context**: `"A cat"`

---

#### Step 1 — Draft model generates 4 tokens

| Step | Draft distribution q(x \| …) | Sampled token |
|---|---|---|
| t=1 | `[0.08, 0.05, 0.70, 0.12, 0.05]` | **sat** (q=0.70) |
| t=2 | `[0.08, 0.05, 0.04, 0.75, 0.08]` | **on**  (q=0.75) |
| t=3 | `[0.06, 0.03, 0.05, 0.05, 0.81]` | **mat** (q=0.81) |
| t=4 | `[0.09, 0.04, 0.04, 0.06, 0.77]` | **the** (q=0.77) — speculative only |

Draft sequence: `["sat", "on", "mat", "the"]`

---

#### Step 2 — Target model scores all 4 in one pass

Input to target: `["A", "cat", "sat", "on", "mat", "the"]`

The target runs a single causal forward pass. At each output position it produces the
distribution for the *next* token:

| Position | Target distribution p(x \| …) | Checks draft token |
|---|---|---|
| after "A cat"      | `[0.06, 0.04, 0.75, 0.10, 0.05]` | t=1 "sat" |
| after "… sat"      | `[0.05, 0.04, 0.03, 0.80, 0.08]` | t=2 "on"  |
| after "… on"       | `[0.10, 0.05, 0.05, 0.30, 0.50]` | t=3 "mat" |
| after "… mat"      | `[0.03, 0.02, 0.02, 0.04, 0.89]` | t=4 "the" |
| after "… the"      | `[0.88, 0.04, 0.03, 0.03, 0.02]` | **bonus token** |

---

#### Step 3 — Rejection sampling at each position

Accept token `x̃ᵢ` with probability `min(1, p(x̃ᵢ) / q(x̃ᵢ))`.

**Token 1: x̃₁ = "sat"**

```
q("sat") = 0.70,  p("sat") = 0.75
acceptance prob = min(1, 0.75 / 0.70) = min(1, 1.071) = 1.0

Draw u ~ Uniform(0,1).  u = 0.43 ≤ 1.0  →  ACCEPT ✓
```

Target assigns *more* probability to "sat" than the draft — always accept (ratio > 1).

**Token 2: x̃₂ = "on"**

```
q("on") = 0.75,  p("on") = 0.80
acceptance prob = min(1, 0.80 / 0.75) = min(1, 1.067) = 1.0

Draw u = 0.61 ≤ 1.0  →  ACCEPT ✓
```

**Token 3: x̃₃ = "mat"**

```
q("mat") = 0.81,  p("mat") = 0.50
acceptance prob = min(1, 0.50 / 0.81) = min(1, 0.617) = 0.617

Draw u = 0.73.  0.73 > 0.617  →  REJECT ✗
```

The target assigns only 50% probability to "mat" but the draft gave it 81% — the draft was
overconfident, so the token is rejected with probability 1 − 0.617 = 0.383.

---

#### Step 4 — Computing the adjusted distribution after a rejection

When token `x̃ᵢ` is rejected, sample a replacement from the **adjusted distribution**:

```
p'(x) = max(0, p(x) − q(x)) / Z
```

This normalised residual removes probability mass where the draft over-estimated and keeps only
the mass where the draft under-estimated.

Using `p(x | "A cat sat on") = [0.10, 0.05, 0.05, 0.30, 0.50]`  
and `q(x | "A cat sat on") = [0.06, 0.03, 0.05, 0.05, 0.81]`:

| Token | p(x) | q(x) | p − q | max(0, p−q) |
|---|---|---|---|---|
| the | 0.10 | 0.06 | +0.04 | **0.04** |
| cat | 0.05 | 0.03 | +0.02 | **0.02** |
| sat | 0.05 | 0.05 |  0.00 | **0.00** |
| on  | 0.30 | 0.05 | +0.25 | **0.25** |
| mat | 0.50 | 0.81 | −0.31 | **0.00** |

```
Z = 0.04 + 0.02 + 0.00 + 0.25 + 0.00 = 0.31

p'(the) = 0.04 / 0.31 = 0.129
p'(cat) = 0.02 / 0.31 = 0.065
p'(sat) = 0.000
p'(on)  = 0.25 / 0.31 = 0.806
p'(mat) = 0.000
```

**"mat" gets zero probability** — the draft was overconfident about it, so all its residual
goes to other tokens. **"on" dominates** because the target preferred it far more than the draft.

Draft tokens x̃₄ = "the" is also discarded (it was conditioned on the wrong x̃₃).

**Round result**: `["sat", "on", <sample from p'>]` = 3 tokens from 1 target pass + 3 draft passes.

---

#### The bonus token (all-accepted case)

When all K=4 tokens pass rejection, the target model's output at position K+1 (after the last
draft token) is already computed at no extra cost. Sample one token from it. This **bonus token**
accounts for the `+1` in the speedup formula.

---

## 3. Why Does Rejection Sampling Preserve the Target Distribution?

We want to prove that for any token x at any position:

```
P(output = x) = p(x)
```

regardless of what the draft model proposes.

### Proof

Every output token x can arrive via exactly two paths.

**Path A** — draft proposes x and we accept it:

```
P(path A) = q(x) · min(1, p(x)/q(x))
           = min(q(x), p(x))
```

**Path B** — draft proposes some y ≠ x, we reject y, then sample x from p'.

First derive the total rejection probability. Let `α = Σ_y min(q(y), p(y))`. Then:

```
P(reject) = Σ_y q(y) · max(0, 1 − p(y)/q(y))
           = Σ_y max(0, q(y) − p(y))
           = 1 − α
```

When we do reject, we sample from `p'(x) = max(0, p(x)−q(x)) / (1−α)`, so:

```
P(path B) = (1−α) · p'(x)
           = (1−α) · max(0, p(x)−q(x)) / (1−α)
           = max(0, p(x)−q(x))
```

**Total probability:**

```
P(output = x) = min(q(x), p(x))  +  max(0, p(x)−q(x))
```

**Case 1 — p(x) ≥ q(x):**

```
min(q(x), p(x)) = q(x)
max(0, p(x)−q(x)) = p(x) − q(x)

P(output = x) = q(x) + p(x) − q(x) = p(x)  ✓
```

**Case 2 — p(x) < q(x):**

```
min(q(x), p(x)) = p(x)
max(0, p(x)−q(x)) = 0

P(output = x) = p(x) + 0 = p(x)  ✓
```

In both cases `P(output = x) = p(x)`. The identity holds for all x, so the output distribution
is exactly the target distribution.

### Intuition for each case

- **p(x) > q(x)** (draft underestimates x): We always accept when x is drafted (ratio > 1),
  and also pick up extra probability via Path B when the draft proposes an overconfident token.
  The two contributions sum exactly to p(x).

- **p(x) < q(x)** (draft overestimates x): We accept with probability p(x)/q(x), which
  downweights the draft's excess down to exactly p(x). Path B contributes zero for x
  (it has no residual mass to give).

The algorithm surgically removes probability from tokens the draft over-estimated and
redistributes it to tokens it under-estimated, leaving each marginal equal to p.

---

## 4. What Determines the Speedup?

### Setting up the formula

Define:

| Symbol | Meaning |
|---|---|
| β | Per-token acceptance rate (probability each draft token is accepted) |
| K | Number of draft tokens per round |
| T_draft | Time for one draft-model forward pass |
| T_target | Time for one target-model forward pass |
| c | Cost ratio: T_draft / T_target |

**Time per speculation round:**

```
Time = K · T_draft  +  T_target
```

The K draft passes are sequential (each token depends on the previous). The single target pass
processes all K tokens in parallel, costing roughly T_target (memory-bound, so weight-loading
dominates — the extra K positions add negligible time for small K).

### Expected tokens per round

Acceptance is sequential — a rejection at position j means tokens j+1…K are discarded.
Total output per round is always (accepted tokens) + 1.

```
E[tokens] = Σ_{j=0}^{K} β^j
           = (1 − β^{K+1}) / (1 − β)
```

Verification at K=1: `E = 1 + β`. With prob (1−β) reject → 1 token; with prob β accept → 2
tokens. So E = 1·(1−β) + 2·β = 1 + β. ✓

For K → ∞ with β close to 1: `E[tokens] ≈ 1 / (1−β)`.

### The speedup formula

Vanilla autoregressive produces 1 token per T_target. Speculative decoding produces E[tokens]
per round in time `K·T_draft + T_target`. Let `C_draft = K · T_draft` be the total draft cost
per round.

```
┌────────────────────────────────────────────────────────────┐
│                                                            │
│            avg_accepted + 1                                │
│   S  =  ──────────────────────                             │
│            1 + C_draft / T_target                          │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### Concrete speedup numbers

**β=0.8, K=4, c=0.1 (draft 10× cheaper than target):**

```
E[tokens] = (1 − 0.8⁵) / (1 − 0.8)
           = (1 − 0.328) / 0.2  =  3.36 tokens per round

C_draft / T_target = 4 × 0.1 = 0.4

Speedup = 3.36 / (1 + 0.4) ≈ 2.4×
```

**β=0.5, K=4, c=0.1:**

```
E[tokens] = (1 − 0.5⁵) / 0.5  ≈  1.94

Speedup = 1.94 / 1.4 ≈ 1.38×
```

**β=0.3, K=4, c=0.1:**

```
E[tokens] = (1 − 0.3⁵) / 0.7  ≈  1.42

Speedup = 1.42 / 1.4 ≈ 1.01×   (essentially no gain)
```

### Speedup across the design space

```
                     β (per-token acceptance rate)
                  0.3     0.5     0.7     0.9
               ┌──────────────────────────────────
  c=0.05  K=4  │  1.2×   1.6×   2.1×   2.8×
  c=0.10  K=4  │  1.0×   1.4×   1.9×   2.5×
  c=0.20  K=4  │  0.8×   1.1×   1.6×   2.1×
  c=0.10  K=8  │  1.0×   1.3×   2.0×   2.9×
```

### When speculative decoding hurts

**1. Low acceptance rate.**  
If β is too low, avg_accepted is near 0 and speedup < 1. The break-even point is roughly:

```
β_min ≈ c · K / (1 + c · K)
```

With K=4, c=0.1: β_min ≈ 0.29. Below that, speculative decoding is slower than baseline.

**2. Draft model too expensive.**  
The draft model should cost roughly 5–20× less than the target. If c=0.5 (draft costs half
the target) and K=4, the denominator is 1 + 2.0 = 3.0, and you need β > 0.6 just to break even.

**3. K too large for the given β.**  
Once K >> 1/(1−β), additional draft tokens have near-zero acceptance probability and you are
paying draft cost for nothing. The optimal K satisfies roughly:

```
K_opt ≈ log(ε) / log(β)
```

for some small probability threshold ε (e.g. where β^K < 0.05).

**4. Large-batch inference.**  
At high batch sizes (B ≈ ridge-point / 2N), the target model is already compute-bound.
Speculative decoding does not help the compute bottleneck and forces the target to run on
smaller effective inputs, destroying the batch utilisation you already had.

**5. Short generation sequences.**  
If the output is only 20–30 tokens, the KV-cache management overhead and draft/target
coordination reduce the practical gain from the theoretical maximum.

### Maximum theoretical speedup

As K → ∞ with fixed β and c, the limit is:

```
S_max = 1 / ((1−β) · (1 + c · K))
```

which grows without bound only if c → 0 (a free draft model). In practice, 2–3× wall-clock
speedup with β ≈ 0.7–0.8 and a 10–20× smaller draft model is a realistic target.
