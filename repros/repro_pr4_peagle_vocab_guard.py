"""Before/after repro for the missing P-EAGLE vocab-mapping guard.

Runs unchanged on both `main` and the fix branch. It configures a reduced draft
vocabulary (16 of the target's 32 tokens) without supplying a t2d/d2t pair, shows
what mapping the model falls back to, and reports whether the backend refuses the
configuration or accepts it.

  main       -> accepted silently, the draft can only ever emit target ids 0..15
  fix branch -> rejected with the same guard EAGLE-3 already has
"""

from __future__ import annotations

import torch
from omegaconf import OmegaConf
from transformers import LlamaConfig

from verl_speco.backends.eagle3_trainer_backend import Eagle3TrainerBackend
from verl_speco.backends.peagle_trainer_backend import PEagleTrainerBackend
from verl_speco.models.peagle import LlamaForCausalLMPeagle

TARGET_VOCAB = 32
DRAFT_VOCAB = 16


def target_config() -> LlamaConfig:
    return LlamaConfig(
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=2,
        vocab_size=TARGET_VOCAB,
        max_position_embeddings=64,
    )


def peagle_backend() -> PEagleTrainerBackend:
    return PEagleTrainerBackend(
        OmegaConf.create(
            {
                "rollout": {
                    "drafter": {
                        "model_path": None,
                        "training": {
                            "peagle_num_draft_layers": 1,
                            "peagle_num_aux_hidden_states": 3,
                            "peagle_num_depths": 2,
                            "peagle_draft_vocab_size": DRAFT_VOCAB,
                        },
                    }
                },
                "model": {"path": "/tmp/none"},
            }
        ),
        target_config(),
    )


def guard_verdict(backend) -> str:
    """Whether build_model refuses the unmapped reduced vocabulary."""
    try:
        backend.build_model()
    except Exception as error:  # noqa: BLE001
        message = str(error)
        if "draft_vocab_size differs from target vocab_size" in message:
            return f"REFUSED: {message}"
        return (
            "accepted the vocab config, then failed later for an unrelated reason: "
            f"{type(error).__name__}: {message.splitlines()[0][:90]}"
        )
    return "accepted the vocab config and built the draft"


def main() -> None:
    print(f"target vocab_size       = {TARGET_VOCAB}")
    print(f"peagle_draft_vocab_size = {DRAFT_VOCAB}")
    print("t2d/d2t supplied        = no")
    print()

    backend = peagle_backend()
    draft_config = backend._build_draft_config(None, target_config())
    draft = LlamaForCausalLMPeagle(draft_config)
    selected = torch.nonzero(draft.t2d, as_tuple=False).flatten().tolist()

    print("[fallback mapping the model constructs]")
    print(f"          draft can emit target ids: {selected}")
    print(f"          d2t buffer               : {draft.d2t.tolist()}")
    print("          this is the first N target ids, not a frequency-derived subset")
    print()

    print("[does the backend refuse it?]")
    print(f"          EAGLE-3 : {'yes' if _eagle3_has_guard() else 'no'}")
    print(f"          P-EAGLE : {guard_verdict(peagle_backend())}")
    print()

    print("[verdict]")
    verdict = guard_verdict(peagle_backend())
    if verdict.startswith("REFUSED"):
        print("          RESULT: P-EAGLE refuses the unmapped reduced vocab (fixed)")
    else:
        print("          RESULT: P-EAGLE trains on an arbitrary vocab slice (bug present)")


def _eagle3_has_guard() -> bool:
    import inspect

    source = inspect.getsource(Eagle3TrainerBackend.build_model)
    return "draft_vocab_size differs from target vocab_size" in source


if __name__ == "__main__":
    main()
