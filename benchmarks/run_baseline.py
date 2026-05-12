"""Autoregressive baseline benchmark.

Runs AutoregressiveDecoder over the prompt dataset and writes one JSON
record per (prompt, temperature, run) to a JSONL results file.  After all
runs a summary table is printed to stdout.

Quick start (tiny dataset, fast smoke-test):
    python benchmarks/run_baseline.py \
        --model gpt2-xl \
        --prompts data/prompts_tiny.json \
        --output results/baseline.jsonl

Full benchmark (150 prompts × 2 temps × 5 runs = 1 500 records):
    python benchmarks/run_baseline.py \
        --model gpt2-xl \
        --prompts data/prompts.json \
        --output results/baseline_gpt2xl.jsonl

Generate the prompt dataset first if data/ does not yet exist:
    python -m src.data.prompts --tokenizer gpt2 --output data/prompts.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Reproducibility — set before any torch work
# ---------------------------------------------------------------------------
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
random.seed(42)

from src.data.prompts import PromptDataset
from src.decoding.autoregressive import AutoregressiveDecoder
from src.models.loader import load_model
from src.models.wrapper import ModelWrapper

_TEMPERATURES = [0.0, 0.6]
_NUM_WARMUP = 3


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Autoregressive baseline benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="gpt2-xl", help="HuggingFace model ID")
    p.add_argument(
        "--prompts",
        default="data/prompts_tiny.json",
        help="Path to the prompt JSON file (build with python -m src.data.prompts)",
    )
    p.add_argument(
        "--output",
        default="results/baseline.jsonl",
        help="Destination JSONL file for per-run results",
    )
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--num-runs", type=int, default=5, help="Timed runs per (prompt, temperature)")
    p.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Model weight dtype",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------


def _run_warmup(decoder: AutoregressiveDecoder, wrapper: ModelWrapper, max_new_tokens: int) -> None:
    """Run a few untimed generations to prime CUDA kernel compilation.

    Without warmup, the first few real measurements are inflated by JIT
    compilation and memory allocator overhead that will not recur on subsequent
    calls.  We use the model's EOS token as a trivial single-token prompt.
    """
    print(f"\n[warmup] running {_NUM_WARMUP} warmup generations …", flush=True)
    warm_ids = torch.tensor([[wrapper.eos_token_id or 50256]], device=wrapper.device)
    for i in range(_NUM_WARMUP):
        decoder.generate(warm_ids, max_new_tokens=min(20, max_new_tokens), temperature=0.0)
        print(f"[warmup] {i + 1}/{_NUM_WARMUP} done", flush=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    print("[warmup] complete — GPU memory stats reset.\n", flush=True)


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------


def _run_one(
    decoder: AutoregressiveDecoder,
    wrapper: ModelWrapper,
    raw_text: str,
    temperature: float,
    max_new_tokens: int,
) -> dict:
    """Tokenise, generate, and return a flat result dict."""
    prompt_ids = wrapper.tokenizer.encode(raw_text, return_tensors="pt")
    prompt_ids = prompt_ids.to(wrapper.device)
    result = decoder.generate(prompt_ids, max_new_tokens=max_new_tokens, temperature=temperature)
    return result.to_dict()


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def _print_summary(records: list[dict]) -> None:
    """Print aggregated stats grouped by (domain, temperature)."""
    if not records:
        print("[summary] no records to summarise.")
        return

    # Accumulate per (domain, temperature)
    groups: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    peak_mem_overall = 0.0

    for r in records:
        key = (r["domain"], r["temperature"])
        groups[key]["tokens_per_second"].append(r["tokens_per_second"])
        groups[key]["time_to_first_token"].append(r["time_to_first_token"])
        groups[key]["latency_p95"].append(r["latency_p95"])
        groups[key]["peak_memory_mb"].append(r["peak_memory_mb"])
        peak_mem_overall = max(peak_mem_overall, r["peak_memory_mb"])

    col_w = 18
    h_domain = "domain".ljust(16)
    h_temp = "temp".rjust(6)
    h_tps = "tok/s (mean)".rjust(col_w)
    h_ttft = "TTFT ms (mean)".rjust(col_w)
    h_p95 = "p95 ms (mean)".rjust(col_w)
    h_mem = "peak mem MB".rjust(col_w)

    sep = "─" * (16 + 6 + col_w * 4 + 12)
    print("\n" + sep)
    print(f"  {h_domain}{h_temp}{h_tps}{h_ttft}{h_p95}{h_mem}")
    print(sep)

    for (domain, temp), vals in sorted(groups.items()):
        mean_tps   = sum(vals["tokens_per_second"]) / len(vals["tokens_per_second"])
        mean_ttft  = sum(vals["time_to_first_token"]) / len(vals["time_to_first_token"]) * 1_000
        mean_p95   = sum(vals["latency_p95"]) / len(vals["latency_p95"]) * 1_000
        mean_mem   = sum(vals["peak_memory_mb"]) / len(vals["peak_memory_mb"])
        n          = len(vals["tokens_per_second"])

        print(
            f"  {domain:<16}{temp:>6.1f}"
            f"{mean_tps:>{col_w}.1f}"
            f"{mean_ttft:>{col_w}.1f}"
            f"{mean_p95:>{col_w}.1f}"
            f"{mean_mem:>{col_w}.0f}"
            f"   (n={n})"
        )

    print(sep)
    print(f"  Peak GPU memory across all runs: {peak_mem_overall:.0f} MB")
    print(sep + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()

    # ── Validate inputs ───────────────────────────────────────────────────────
    prompts_path = Path(args.prompts)
    if not prompts_path.exists():
        print(
            f"[error] prompt file not found: {prompts_path}\n"
            f"        Generate it first:\n"
            f"          python -m src.data.prompts --tokenizer gpt2 --output {prompts_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"[baseline] loading model: {args.model}  dtype={args.dtype}")
    model, tokenizer = load_model(args.model, args.dtype)
    device = next(model.parameters()).device
    wrapper = ModelWrapper(model, tokenizer, device)
    decoder = AutoregressiveDecoder(wrapper)
    print(f"[baseline] model ready — {wrapper}\n")

    # ── Load prompts ──────────────────────────────────────────────────────────
    dataset = PromptDataset.load(prompts_path)
    prompts = dataset.get_all()
    print(
        f"[baseline] {len(prompts)} prompts  ×  "
        f"{len(_TEMPERATURES)} temperatures  ×  "
        f"{args.num_runs} runs  =  "
        f"{len(prompts) * len(_TEMPERATURES) * args.num_runs} total generations\n"
    )

    # ── Warmup ────────────────────────────────────────────────────────────────
    _run_warmup(decoder, wrapper, args.max_new_tokens)

    # ── Benchmark loop ────────────────────────────────────────────────────────
    all_records: list[dict] = []
    total = len(prompts) * len(_TEMPERATURES) * args.num_runs
    done = 0
    t_wall_start = time.perf_counter()

    with open(output_path, "w", encoding="utf-8") as out_f:
        for prompt in prompts:
            for temperature in _TEMPERATURES:
                for run_idx in range(args.num_runs):
                    result_dict = _run_one(
                        decoder,
                        wrapper,
                        prompt.raw_text,
                        temperature,
                        args.max_new_tokens,
                    )

                    record = {
                        "model_id": args.model,
                        "prompt_id": prompt.prompt_id,
                        "domain": prompt.domain,
                        "prompt_token_count": prompt.token_count,
                        "temperature": temperature,
                        "max_new_tokens": args.max_new_tokens,
                        "run_idx": run_idx,
                        **result_dict,
                    }

                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_f.flush()
                    all_records.append(record)

                    done += 1
                    elapsed = time.perf_counter() - t_wall_start
                    remaining_est = (elapsed / done) * (total - done)
                    print(
                        f"[{done:>4}/{total}]  {prompt.prompt_id:<18}  "
                        f"temp={temperature:.1f}  run={run_idx}  "
                        f"{result_dict['tokens_per_second']:>6.1f} tok/s  "
                        f"ETA {remaining_est:>5.0f}s",
                        flush=True,
                    )

    # ── Summary ───────────────────────────────────────────────────────────────
    wall_total = time.perf_counter() - t_wall_start
    print(f"\n[baseline] all runs complete in {wall_total:.1f}s")
    print(f"[baseline] results written to {output_path}  ({len(all_records)} records)")
    _print_summary(all_records)


if __name__ == "__main__":
    main()
