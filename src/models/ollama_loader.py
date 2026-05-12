"""Ollama convenience wrapper for quick hardware validation.

PURPOSE
-------
This module exists for one reason: quickly confirm that a model runs on your
hardware before spending time downloading multi-gigabyte HuggingFace weights.
It is a development convenience tool, not part of the benchmarking pipeline.

Example workflow::

    # 1. Is the model small enough to fit on my GPU?
    python -m src.models.ollama_loader --model llama3:8b --prompt "Hello"

    # 2. Yes — now download the full HF weights for real benchmarking.
    python -m src.models.loader --model meta-llama/Meta-Llama-3-8B-Instruct --dtype float16

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY OLLAMA CANNOT BE USED FOR SPECULATIVE DECODING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Speculative decoding requires four capabilities that Ollama's HTTP API does
not expose:

1. FULL VOCABULARY DISTRIBUTIONS AT EVERY POSITION
   The rejection-sampling step (see docs/speculative_decoding_theory.md §3)
   requires p_target(x) for every token x in the vocabulary at every draft
   position — i.e., the raw logit vector of shape [vocab_size] before any
   sampling.  Ollama returns only the single sampled token (and optionally a
   small top-k list via `options.logprobs`).  A truncated top-k distribution
   cannot be used for provably correct rejection sampling: the normalisation
   constant for the adjusted distribution p'(x) = max(0, p-q) / Z requires
   mass over the full vocabulary.

2. PARALLEL SCORING OF MULTIPLE POSITIONS IN ONE FORWARD PASS
   After the draft model proposes K tokens, the target model must score all K
   positions simultaneously in a single causal forward pass — treating the
   draft sequence as a batch.  This is what makes speculative decoding fast:
   one target forward pass instead of K sequential ones.  Ollama's /api/generate
   endpoint is a sequential autoregressive loop with no way to inject a
   pre-formed token sequence and retrieve per-position logits.

3. KV CACHE ACCESS AND MANIPULATION
   When a draft token is rejected at position j, all subsequent KV cache
   entries (positions j+1 … K) must be discarded so the next speculative
   round starts from the correct state.  Ollama manages its KV cache
   internally and provides no API surface for truncation or inspection.

4. DRAFT / TARGET MODEL COORDINATION IN SHARED MEMORY
   Efficient speculative decoding passes tensors directly between the draft
   and target models on-GPU, avoiding serialisation to JSON and HTTP round
   trips.  The latency of two HTTP calls per speculation round would dwarf
   any throughput gain.

THE CORRECT TOOL: transformers.AutoModelForCausalLM
   The HuggingFace Transformers library gives direct access to logit tensors
   (model(...).logits), full control over the KV cache (past_key_values /
   use_cache), and the ability to run an arbitrary-length input in one
   forward pass.  See src/models/loader.py and src/decoding/speculative.py.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from dataclasses import dataclass

_BASE_URL = "http://localhost:11434"
_TIMEOUT = 5  # seconds for health-check and listing requests


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


@dataclass
class OllamaModelInfo:
    """Metadata for a single model returned by the Ollama /api/tags endpoint.

    Attributes:
        name:                Full model tag, e.g. ``"llama3:8b"``.
        parameter_size:      Human-readable size string, e.g. ``"8B"``.
        quantization_level:  GGUF quantisation level, e.g. ``"Q4_0"``.
        size_gb:             On-disk size in gigabytes.
    """

    name: str
    parameter_size: str
    quantization_level: str
    size_gb: float


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------


def is_running() -> bool:
    """Return True if an Ollama server is reachable at localhost:11434.

    Makes a lightweight GET request to the root endpoint; a 200 response
    (or any HTTP response, really) means the daemon is up.  A connection
    error or timeout means it is not.
    """
    try:
        with urllib.request.urlopen(_BASE_URL, timeout=_TIMEOUT):
            return True
    except urllib.error.URLError:
        return False
    except OSError:
        return False


def _get(path: str, timeout: int = _TIMEOUT) -> dict:
    """Make a GET request to the Ollama API and return the parsed JSON body.

    Args:
        path:    URL path, e.g. ``"/api/tags"``.
        timeout: Socket timeout in seconds.

    Raises:
        RuntimeError: If the server is unreachable or returns a non-200 status.
    """
    url = _BASE_URL + path
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Cannot reach Ollama at {_BASE_URL}. "
            "Is the daemon running?  Try: ollama serve"
        ) from exc


def _post(path: str, payload: dict, timeout: int = 120) -> dict:
    """POST JSON payload to the Ollama API and return the parsed response.

    Args:
        path:    URL path, e.g. ``"/api/generate"``.
        payload: Request body as a Python dict (will be JSON-encoded).
        timeout: Socket timeout in seconds.  Use a generous value for
                 generation requests on large models.

    Raises:
        RuntimeError: If the server is unreachable or returns an error status.
    """
    url = _BASE_URL + path
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"Ollama API error {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Cannot reach Ollama at {_BASE_URL}. "
            "Is the daemon running?  Try: ollama serve"
        ) from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_models() -> list[OllamaModelInfo]:
    """Return metadata for every model currently available in Ollama.

    Calls GET /api/tags and extracts name, parameter size, quantisation
    level, and on-disk size.

    Returns:
        List of :class:`OllamaModelInfo`, one per available model.

    Raises:
        RuntimeError: If Ollama is not running.
    """
    data = _get("/api/tags")
    models: list[OllamaModelInfo] = []
    for entry in data.get("models", []):
        details = entry.get("details", {})
        models.append(
            OllamaModelInfo(
                name=entry.get("name", "unknown"),
                parameter_size=details.get("parameter_size", "?"),
                quantization_level=details.get("quantization_level", "?"),
                size_gb=entry.get("size", 0) / 1024**3,
            )
        )
    return models


def generate(
    model: str,
    prompt: str,
    max_tokens: int = 128,
    temperature: float = 0.0,
    timeout: int = 120,
) -> str:
    """Generate a text continuation from an Ollama-hosted model.

    This is a convenience wrapper for quick smoke tests.  Do NOT use it
    inside the speculative decoding pipeline — see the module docstring
    for a detailed explanation of why.

    Args:
        model:       Ollama model tag, e.g. ``"llama3:8b"`` or ``"gpt2"``.
        prompt:      Input text to continue.
        max_tokens:  Maximum number of tokens to generate.
        temperature: Sampling temperature.  0.0 → greedy (deterministic).
        timeout:     HTTP timeout in seconds.  Increase for large models
                     or long outputs on slow hardware.

    Returns:
        Generated text (the continuation only, not the prompt).

    Raises:
        RuntimeError: If Ollama is not running or the model is not available.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature,
        },
    }
    response = _post("/api/generate", payload, timeout=timeout)
    return response.get("response", "")


def check_and_print_status() -> bool:
    """Print a human-readable status report and return True if Ollama is up.

    Checks connectivity, then lists all available models with their sizes
    and quantisation levels.  Useful as a first step before running any
    generation.

    Returns:
        True if Ollama is reachable, False otherwise.
    """
    print(f"[ollama] checking server at {_BASE_URL} …")
    if not is_running():
        print("[ollama] ✗ server not reachable.")
        print("         Start it with:  ollama serve")
        print("         Install from:   https://ollama.com")
        return False

    print("[ollama] ✓ server is running.\n")

    models = list_models()
    if not models:
        print("[ollama] no models installed yet.")
        print("         Pull one with:  ollama pull llama3:8b")
        return True

    col = "{:<35} {:>8}  {:>6}  {:>8}"
    print(col.format("model", "params", "quant", "size (GB)"))
    print("-" * 62)
    for m in models:
        print(col.format(m.name, m.parameter_size, m.quantization_level, f"{m.size_gb:.1f}"))
    print()
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Quick hardware validation via Ollama. "
            "NOT for use in the benchmarking pipeline — see module docstring."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        metavar="TAG",
        default=None,
        help=(
            "Ollama model tag to generate from (e.g. llama3:8b). "
            "Omit to only check server status and list models."
        ),
    )
    parser.add_argument(
        "--prompt",
        metavar="TEXT",
        default="The quick brown fox",
        help="Prompt string to pass to the model.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        metavar="N",
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 = greedy).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    running = check_and_print_status()
    if not running:
        return

    if args.model is None:
        return

    print(f"[ollama] model    : {args.model}")
    print(f"[ollama] prompt   : {args.prompt!r}")
    print(f"[ollama] generating {args.max_tokens} tokens …\n")

    output = generate(
        model=args.model,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    print(f"[ollama] output:\n{output}")
    print(
        "\n[ollama] NOTE: this output was produced by Ollama's sampler. "
        "For speculative decoding benchmarks, load the equivalent HuggingFace "
        "model via:  python -m src.models.loader --model <hf_id> --dtype float16"
    )


if __name__ == "__main__":
    main()
