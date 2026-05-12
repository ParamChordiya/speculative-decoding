"""Model loading utilities.

Single entry-point for loading any causal LM supported by HuggingFace
Transformers, with optional bitsandbytes 4-bit quantisation.  Every caller
should go through :func:`load_model` so that dtype handling and post-load
reporting stay consistent across the codebase.

CLI usage::

    python -m src.models.loader --model gpt2 --dtype float16
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Map string dtype names to torch scalar types.  "int4" is handled separately
# because it is not a native torch dtype — it requires a BitsAndBytesConfig.
_DTYPE_MAP: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_load_kwargs(dtype: str) -> dict:
    """Translate a dtype string into keyword arguments for from_pretrained.

    For "int4" this produces a BitsAndBytesConfig that enables NF4
    quantisation with double-quant compression and float16 compute kernels —
    the standard configuration for inference-quality 4-bit loading.
    """
    if dtype == "int4":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        return {"quantization_config": bnb_config, "device_map": "auto"}

    if dtype not in _DTYPE_MAP:
        valid = list(_DTYPE_MAP) + ["int4"]
        raise ValueError(f"Unsupported dtype {dtype!r}. Choose from: {valid}")

    return {"torch_dtype": _DTYPE_MAP[dtype], "device_map": "auto"}


def _print_model_info(model: torch.nn.Module, model_id: str) -> None:
    """Print parameter count and on-device memory footprint after loading."""
    n_params = sum(p.numel() for p in model.parameters())
    mem_bytes = model.get_memory_footprint()
    print(f"  model      : {model_id}")
    print(f"  parameters : {n_params / 1e6:.1f} M")
    print(f"  memory     : {mem_bytes / 1024 ** 3:.2f} GB")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_model(
    model_id: str,
    dtype: str,
) -> tuple[torch.nn.Module, AutoTokenizer]:
    """Load a causal LM and its tokenizer from HuggingFace.

    Uses ``device_map="auto"`` so the model is distributed across all
    available GPUs (or falls back to CPU/MPS if no CUDA device is present).
    The model is set to eval mode before being returned.

    Args:
        model_id: HuggingFace model identifier, e.g. ``"gpt2"`` or
                  ``"meta-llama/Meta-Llama-3-8B-Instruct"``.
        dtype:    Weight precision — one of ``"float16"``, ``"bfloat16"``,
                  ``"float32"``, or ``"int4"``.  ``"int4"`` enables
                  bitsandbytes NF4 quantisation with float16 compute kernels.

    Returns:
        ``(model, tokenizer)`` — both ready for inference.
    """
    print(f"[loader] tokenizer  ← {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Many causal LMs have no pad token; set it to eos so batched encoding
    # works without errors.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[loader] model      ← {model_id}  (dtype={dtype})")
    load_kwargs = _build_load_kwargs(dtype)
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model.eval()

    _print_model_info(model, model_id)
    return model, tokenizer


@torch.inference_mode()
def _verify_generation(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    n_tokens: int = 10,
) -> str:
    """Generate a short continuation to confirm end-to-end inference works.

    Greedy decoding is used so the output is deterministic and easy to
    inspect by eye.  This function is intentionally simple — it is a smoke
    test, not a benchmark.
    """
    prompt = "The quick brown fox"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=n_tokens,
        do_sample=False,
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a HuggingFace causal LM and run a short generation check.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        required=True,
        metavar="MODEL_ID",
        help="HuggingFace model ID (e.g. gpt2, gpt2-xl).",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32", "int4"],
        help="Weight dtype to load the model in.",
    )
    parser.add_argument(
        "--n-tokens",
        type=int,
        default=10,
        metavar="N",
        help="Number of tokens to generate during the verification step.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    model, tokenizer = load_model(args.model, args.dtype)

    print(f"\n[loader] generating {args.n_tokens} tokens …")
    output = _verify_generation(model, tokenizer, n_tokens=args.n_tokens)
    print(f"[loader] output: {output!r}")
    print("[loader] OK")


if __name__ == "__main__":
    main()
