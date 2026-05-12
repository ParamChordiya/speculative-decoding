from dataclasses import dataclass
from typing import Literal

DType = Literal["float16", "bfloat16", "float32", "int4"]


@dataclass(frozen=True)
class ModelPair:
    """A (draft, target) pair for speculative decoding.

    The draft model generates candidate tokens cheaply; the target model
    verifies them in a single parallel forward pass.  Both models must share
    the same tokenizer vocabulary for rejection sampling to be valid.

    Attributes:
        pair_name:       Short identifier used throughout configs and results.
        draft_model_id:  HuggingFace model ID for the small draft model.
        target_model_id: HuggingFace model ID for the large target model.
        draft_dtype:     Weight precision to load the draft model with.
        target_dtype:    Weight precision to load the target model with.
    """

    pair_name: str
    draft_model_id: str
    target_model_id: str
    draft_dtype: DType
    target_dtype: DType


# ---------------------------------------------------------------------------
# Registered pairs
# ---------------------------------------------------------------------------

_PAIRS: tuple[ModelPair, ...] = (
    # Pure development pair — both models are GPT-2 family so they share a
    # vocabulary, load fast, and need no special access tokens.
    ModelPair(
        pair_name="gpt2_dev",
        draft_model_id="gpt2",
        target_model_id="gpt2-xl",
        draft_dtype="float16",
        target_dtype="float16",
    ),
    # TinyLlama (1.1 B) drafting for Llama-3-8B.  Shared BPE vocabulary
    # (32 000 tokens) makes acceptance rates reasonable despite the
    # architecture mismatch.
    ModelPair(
        pair_name="tinyllama_llama3",
        draft_model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        target_model_id="meta-llama/Meta-Llama-3-8B-Instruct",
        draft_dtype="float16",
        target_dtype="float16",
    ),
    # Phi-3-mini (3.8 B) drafting for Llama-3-8B.  Higher draft cost than
    # TinyLlama but expected to yield a better acceptance rate due to the
    # closer capability level.
    ModelPair(
        pair_name="phi3_llama3",
        draft_model_id="microsoft/Phi-3-mini-4k-instruct",
        target_model_id="meta-llama/Meta-Llama-3-8B-Instruct",
        draft_dtype="float16",
        target_dtype="float16",
    ),
    # Self-speculation: the same 8B model in two precision levels.  The
    # 4-bit quantised copy acts as the draft; the full float16 copy is the
    # target.  Acceptance rate is near-perfect by construction — speedup
    # comes solely from the reduced memory bandwidth of the quantised draft.
    ModelPair(
        pair_name="llama3_self",
        draft_model_id="meta-llama/Meta-Llama-3-8B-Instruct",
        target_model_id="meta-llama/Meta-Llama-3-8B-Instruct",
        draft_dtype="int4",
        target_dtype="float16",
    ),
)

REGISTRY: dict[str, ModelPair] = {pair.pair_name: pair for pair in _PAIRS}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_pair(name: str) -> ModelPair:
    """Return the ModelPair registered under *name*.

    Raises:
        KeyError: if *name* is not in the registry, with a message listing
                  the available names.
    """
    if name not in REGISTRY:
        available = ", ".join(REGISTRY)
        raise KeyError(f"Unknown pair {name!r}. Available pairs: {available}")
    return REGISTRY[name]


def list_pairs() -> list[str]:
    """Return the names of all registered model pairs in definition order."""
    return list(REGISTRY)
