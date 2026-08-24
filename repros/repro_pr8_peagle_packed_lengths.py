"""Before/after repro for the P-EAGLE packed-document-length handling.

Runs unchanged on both `main` and the fix branch. `seq_lengths` is a flat list of
document lengths for ONE packed sequence, but the forward loops over the batch and
hands the whole tensor to every row, and never checks that the lengths cover the
sequence.

  main       -> both malformed inputs are accepted and drafted on
  fix branch -> both are rejected with a specific message
"""

from __future__ import annotations

import torch

from verl_speco.backends.peagle_trainer_backend import PEagleTrainingModel
from verl_speco.models.peagle import LlamaForCausalLMPeagle, PeagleConfig


def tiny_config() -> PeagleConfig:
    return PeagleConfig(
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=2,
        num_draft_layers=2,
        target_hidden_size=8,
        num_aux_hidden_states=3,
        vocab_size=32,
        draft_vocab_size=32,
        num_depths=4,
        mask_token_id=31,
        pad_token_id=0,
        rms_norm_eps=1e-6,
        max_position_embeddings=64,
    )


def run(batch_size: int, seq_len: int, seq_lengths: torch.Tensor) -> str:
    torch.manual_seed(0)
    config = tiny_config()
    model = PEagleTrainingModel(
        LlamaForCausalLMPeagle(config), num_depths=2, down_sample_ratio=0.5
    )
    try:
        model(
            input_ids=torch.zeros(batch_size, seq_len, dtype=torch.long),
            aux_hidden=torch.zeros(
                batch_size, seq_len, config.target_hidden_size * 3
            ),
            loss_mask=torch.ones(batch_size, seq_len),
            attention_mask=torch.ones(batch_size, seq_len, dtype=torch.long),
            target_logits=torch.zeros(batch_size, seq_len, config.vocab_size),
            seq_lengths=seq_lengths,
        )
    except ValueError as error:
        return f"REJECTED: {error}"
    except Exception as error:  # noqa: BLE001
        return f"failed for another reason: {type(error).__name__}: {error}"
    return "ACCEPTED (drafted on a mismatched document layout)"


def main() -> None:
    cases = [
        (
            "batch of 2 rows, seq_lengths=[4, 4] for an 8-token sequence",
            2,
            8,
            torch.tensor([4, 4]),
        ),
        (
            "1 row of 8 tokens, seq_lengths=[3, 2] covers only 5",
            1,
            8,
            torch.tensor([3, 2]),
        ),
    ]

    rejected = 0
    for label, batch_size, seq_len, seq_lengths in cases:
        outcome = run(batch_size, seq_len, seq_lengths)
        rejected += outcome.startswith("REJECTED")
        print(f"[{label}]")
        print(f"          {outcome}")
        print()

    print("[verdict]")
    if rejected == len(cases):
        print("          RESULT: malformed packed lengths are rejected (fixed)")
    else:
        print(f"          RESULT: {len(cases) - rejected} of {len(cases)} accepted silently (bug present)")


if __name__ == "__main__":
    main()
