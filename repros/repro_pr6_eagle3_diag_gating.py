"""Before/after repro for the EAGLE-3 drafter quality diagnostics gating.

Runs unchanged on both `main` and the fix branch. It drives `compute_loss` with a
stub draft at INFO and at DEBUG, counting how many `Tensor.item()` device syncs
each pays for and what the diagnostics log at.

  main       -> the diagnostics run at every level and the summary is a WARNING
  fix branch -> above DEBUG they are skipped entirely, and the summary is a DEBUG
"""

from __future__ import annotations

import logging

import torch
from omegaconf import OmegaConf

from verl_speco.backends.eagle3_trainer_backend import Eagle3TrainerBackend

BACKEND_LOGGER = "verl_speco.backends.eagle3_trainer_backend"
QUALITY_LOG_PREFIX = "[drafter logits quality]"

VOCAB, SEQ_LEN, TTT_LENGTH = 6, 4, 2


class _FakeDraft:
    t2d = torch.ones(VOCAB, dtype=torch.bool)

    def __call__(self, **kwargs):
        return {
            "logits": [torch.randn(1, SEQ_LEN, VOCAB) for _ in range(TTT_LENGTH)],
            "position_masks": [torch.ones(1, SEQ_LEN) for _ in range(TTT_LENGTH)],
        }


class _Recorder(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)


def build_case():
    backend = Eagle3TrainerBackend(
        OmegaConf.create(
            {
                "rollout": {
                    "drafter": {
                        "training": {"use_logits": False, "ttt_length": TTT_LENGTH}
                    }
                },
                "model": {"path": "/tmp/none"},
            }
        ),
        OmegaConf.create({}),
    )
    backend.target_model = lambda last_hidden: torch.zeros(
        *last_hidden.shape[:-1], VOCAB
    )
    batch = {
        "input_ids": torch.zeros(1, SEQ_LEN, dtype=torch.long),
        "hidden_states": torch.zeros(1, SEQ_LEN, 8),
        "last_hidden_states": torch.zeros(1, SEQ_LEN, 8),
        "attention_mask": torch.ones(1, SEQ_LEN, dtype=torch.long),
        "loss_mask": torch.ones(1, SEQ_LEN),
        "position_ids": torch.arange(SEQ_LEN).unsqueeze(0),
    }
    return backend, _FakeDraft(), batch


def measure(level: int):
    """Returns (item_call_count, quality_record_level_name_or_none)."""
    backend, model, batch = build_case()
    backend_logger = logging.getLogger(BACKEND_LOGGER)
    backend_logger.setLevel(level)
    recorder = _Recorder()
    backend_logger.addHandler(recorder)

    original_item = torch.Tensor.item
    calls = 0

    def counting_item(self):
        nonlocal calls
        calls += 1
        return original_item(self)

    torch.Tensor.item = counting_item
    try:
        backend.compute_loss(model, batch, 0)
    finally:
        torch.Tensor.item = original_item
        backend_logger.removeHandler(recorder)

    quality = [r for r in recorder.records if QUALITY_LOG_PREFIX in r.getMessage()]
    return calls, (quality[0].levelname if quality else None)


def main() -> None:
    torch.manual_seed(0)
    info_syncs, info_level = measure(logging.INFO)
    debug_syncs, debug_level = measure(logging.DEBUG)

    print("[one compute_loss call, ttt_length=2]")
    print(f"          at INFO : {info_syncs} Tensor.item() syncs, quality log = {info_level}")
    print(f"          at DEBUG: {debug_syncs} Tensor.item() syncs, quality log = {debug_level}")
    print()

    print("[verdict]")
    if info_syncs < debug_syncs and info_level is None:
        print("          diagnostics are skipped above DEBUG")
        print(f"          summary line logs at {debug_level}")
        print("          RESULT: gated on the log level (fixed)")
    else:
        print(f"          diagnostics run at INFO too ({info_syncs} syncs)")
        print(f"          summary line logs at {info_level}")
        print("          RESULT: unconditional syncs on the hot path (bug present)")


if __name__ == "__main__":
    main()
