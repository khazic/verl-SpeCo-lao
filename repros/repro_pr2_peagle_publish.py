"""Before/after repro for the P-EAGLE hot-publish filter.

Runs unchanged on both `main` and the fix branch. It builds a tiny P-EAGLE draft,
runs the real publish filter over its state dict with a real `PEagleTrainerBackend`,
and reports which trained tensors actually reach the rollout engine.

  main       -> lm_head.weight and embed_tokens.weight are dropped, so the engine
                keeps the initial head and embedding forever
  fix branch -> both are published alongside the rest of the draft
"""

from __future__ import annotations

import inspect

import torch
from omegaconf import OmegaConf

from verl_speco.backends.dflash_trainer_backend import DFlashTrainerBackend
from verl_speco.backends.domino_trainer_backend import DominoTrainerBackend
from verl_speco.backends.dspark_trainer_backend import DSparkTrainerBackend
from verl_speco.backends.eagle1_trainer_backend import Eagle1TrainerBackend
from verl_speco.backends.eagle3_trainer_backend import Eagle3TrainerBackend
from verl_speco.backends.peagle_trainer_backend import PEagleTrainerBackend
from verl_speco.models.peagle import LlamaForCausalLMPeagle, PeagleConfig
from verl_speco.trainer.base_trainer import DrafterBaseTrainer

BACKENDS = [
    Eagle3TrainerBackend,
    Eagle1TrainerBackend,
    PEagleTrainerBackend,
    DFlashTrainerBackend,
    DSparkTrainerBackend,
    DominoTrainerBackend,
]


def tiny_peagle_config() -> PeagleConfig:
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


def freezes_embedding(backend_cls) -> bool:
    """Whether the backend's build_model freezes the target-seeded embedding."""
    try:
        source = inspect.getsource(backend_cls.build_model)
    except (OSError, TypeError):
        return False
    return "freeze_embedding" in source


def main() -> None:
    print("[which backends freeze the draft embedding in build_model]")
    for backend_cls in BACKENDS:
        state = "freezes" if freezes_embedding(backend_cls) else "TRAINS IT"
        print(f"          {backend_cls.__name__:<24} {state}")
    print()

    torch.manual_seed(0)
    config = tiny_peagle_config()
    draft = LlamaForCausalLMPeagle(config)

    print("[trained parameters on the P-EAGLE draft]")
    for name in ("embed_tokens.weight", "lm_head.weight", "fc.weight", "mask_hidden"):
        parameter = dict(draft.named_parameters())[name]
        print(f"          {name:<24} requires_grad={parameter.requires_grad}")
    print()

    backend = PEagleTrainerBackend(
        OmegaConf.create(
            {
                "rollout": {"drafter": {"training": {}, "model_path": None}},
                "model": {"path": "/tmp/none"},
            }
        ),
        OmegaConf.create({}),
    )

    trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
    trainer.backend = backend
    trainer.training_device_mesh = None
    trainer._frozen_param_names = {"model.embed_tokens.weight"}
    trainer.model = draft

    published = set(trainer._get_trainable_state_dict())
    trained = {
        name
        for name, parameter in draft.named_parameters()
        if parameter.requires_grad and torch.is_floating_point(parameter)
    }
    dropped = sorted(trained - published)

    print("[hot publish, P-EAGLE]")
    print(f"          trained tensors    : {len(trained)}")
    print(f"          published tensors  : {len(published & trained)}")
    print(f"          dropped tensors    : {len(dropped)}")
    for name in dropped:
        print(f"                             - {name}")
    print()

    print("[verdict]")
    for name in ("embed_tokens.weight", "lm_head.weight"):
        status = "published" if name in published else "DROPPED"
        print(f"          {name:<24} {status}")
    if dropped:
        print("          RESULT: trained P-EAGLE weights never reach the engine (bug present)")
    else:
        print("          RESULT: every trained P-EAGLE weight is published (fixed)")


if __name__ == "__main__":
    main()
