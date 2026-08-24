"""Before/after repro for the Domino base-anchor curriculum resume bug.

Runs unchanged on both `main` and the fix branch. It simulates a training run
that reaches optimizer step N, then a process restart that resumes from the same
checkpoint step, and reports the curriculum weight and the correction head's
gradient magnitude on both sides of the restart.

  main       -> the curriculum restarts, lambda_base jumps back to ~1.0 and the
                correction head keeps ~1% of its gradient
  fix branch -> the curriculum continues, lambda_base stays 0.0 and the head
                keeps 100% of its gradient
"""

from __future__ import annotations

import torch
from omegaconf import OmegaConf

from verl_speco.backends.domino_trainer_backend import (
    DominoTrainerBackend,
    DominoTrainingModel,
)
from verl_speco.models.domino import DominoConfig, DominoDraftModel

DECAY_STEPS = 100
RESUME_AT_STEP = 100


def tiny_config() -> DominoConfig:
    return DominoConfig(
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=1,
        vocab_size=32,
        num_target_layers=4,
        num_context_layers=2,
        target_hidden_size=8,
        target_num_hidden_layers=4,
        target_layer_ids=[1, 3],
        mask_token_id=31,
        block_size=4,
        num_anchors=8,
        emb_dim=6,
        gru_hidden_dim=10,
        pure_draft_prefix_len=1,
        rms_norm_eps=1e-6,
        max_position_embeddings=64,
    )


def build_wrapper(config: DominoConfig) -> DominoTrainingModel:
    # Same seed on both sides of the restart, so the two wrappers hold identical
    # weights and any difference comes purely from the curriculum weight.
    torch.manual_seed(0)
    return DominoTrainingModel(
        draft_model=DominoDraftModel(config),
        block_size=config.block_size,
        num_anchors=config.num_anchors,
        pure_draft_prefix_len=config.pure_draft_prefix_len,
        lambda_base_start=1.0,
        lambda_base_decay_steps=DECAY_STEPS,
    )


def curriculum_step(model: DominoTrainingModel):
    for attribute in ("_curriculum_step", "_forward_count"):
        if hasattr(model, attribute):
            return getattr(model, attribute)
    return None


def measure(model: DominoTrainingModel, batch) -> tuple[float, float]:
    """One forward/backward; returns (lambda_base, correction-head grad norm)."""
    model.zero_grad(set_to_none=True)
    torch.manual_seed(1)
    outputs = model(*batch)
    loss, diagnostics = outputs[0], outputs[5]
    loss.backward()
    grad_norm = sum(
        float(param.grad.norm())
        for param in model.draft_model.embed_proj.parameters()
        if param.grad is not None
    )
    return float(diagnostics["domino_lambda_base"]), grad_norm


def main() -> None:
    config = tiny_config()

    torch.manual_seed(7)
    batch_size, seq_len = 2, 16
    batch = (
        torch.randint(0, config.vocab_size, (batch_size, seq_len)),
        [
            torch.randn(batch_size, seq_len, config.target_hidden_size)
            for _ in config.target_layer_ids
        ],
        torch.ones(batch_size, seq_len, dtype=torch.long),
        torch.randn(config.vocab_size, config.hidden_size),
    )

    backend = DominoTrainerBackend(
        OmegaConf.create(
            {"rollout": {"drafter": {"training": {}}}, "model": {"path": "/tmp/none"}}
        ),
        OmegaConf.create({}),
    )

    print(f"domino_lambda_base_decay_steps = {DECAY_STEPS}")
    print(f"resume from optimizer step     = {RESUME_AT_STEP}")
    print()

    print(f"[phase 1] fresh run, walk the curriculum to step {RESUME_AT_STEP}")
    model = build_wrapper(config)
    backend.setup_optimizer(model, OmegaConf.create({"lr": 1e-4}))
    for _ in range(RESUME_AT_STEP):
        torch.manual_seed(1)
        model(*batch)
    lambda_before, grad_before = measure(model, batch)
    print(
        f"          curriculum_step={curriculum_step(model)} "
        f"lambda_base={lambda_before:.4f} head_grad_norm={grad_before:.6f}"
    )
    print()

    print(f"[phase 2] process restart, resume the drafter at step {RESUME_AT_STEP}")
    resumed = build_wrapper(config)
    backend.setup_optimizer(
        resumed,
        OmegaConf.create({"lr": 1e-4, "_resume_optimizer_steps": RESUME_AT_STEP}),
    )
    lambda_after, grad_after = measure(resumed, batch)
    print(
        f"          curriculum_step={curriculum_step(resumed)} "
        f"lambda_base={lambda_after:.4f} head_grad_norm={grad_after:.6f}"
    )
    print()

    retained = 100.0 * grad_after / max(grad_before, 1e-12)
    print("[verdict]")
    print(f"          lambda_base across the restart: {lambda_before:.4f} -> {lambda_after:.4f}")
    print(f"          correction-head gradient retained: {retained:.2f}%")
    if lambda_after > lambda_before + 1e-6:
        print("          RESULT: curriculum RESTARTED (bug present)")
    else:
        print("          RESULT: curriculum CONTINUED (fixed)")


if __name__ == "__main__":
    main()
