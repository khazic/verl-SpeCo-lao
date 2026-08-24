# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contract tests for the DFlash2 drafter backend.

CPU-light: they exercise the two DFlash2 modules (the two-tap dynamic
convolution and the candidate selector), the block-locality invariant the
training layout requires, the algorithm routing, and the block-drafter
classification. The full training forward is validated on GPU by
``tests/special_standalone/dflash2_gpu_smoke.py``.
"""

from __future__ import annotations

import pytest


def _tiny_dflash2_config(**overrides):
    from verl_speco.models.dflash2 import DFlash2Config

    kwargs = dict(
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
        conv_kernel_size=2,
        conv_group_size=4,
        selector_rank=6,
        selector_top_k=5,
        rms_norm_eps=1e-6,
        max_position_embeddings=64,
    )
    kwargs.update(overrides)
    return DFlash2Config(**kwargs)


def test_dflash2_model_builds_conv_and_selector() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from verl_speco.models.dflash2 import DFlash2DraftModel

    config = _tiny_dflash2_config()
    model = DFlash2DraftModel(config)

    # Every layer gets a conv around attention and around the MLP.
    for layer in model.layers:
        assert layer.attention_conv is not None
        assert layer.mlp_conv is not None
        assert layer.attention_conv.kernel_size == config.conv_kernel_size
        assert layer.attention_conv.group_size == config.conv_group_size
        groups = config.hidden_size // config.conv_group_size
        assert layer.attention_conv.kernel_projection.out_features == (
            2 * config.conv_kernel_size * groups
        )

    selector = model.candidate_selector
    assert selector.predecessor_codebook.num_embeddings == config.vocab_size
    assert selector.successor_codebook.num_embeddings == config.vocab_size
    assert selector.hidden_projection.in_features == config.hidden_size
    assert selector.hidden_projection.out_features == config.selector_rank


def test_plain_dflash_layers_keep_conv_hooks_inert() -> None:
    """The conv hooks live on the shared DFlash layer; DFlash must not use them."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from verl_speco.models.dflash import DFlashConfig, DFlashDraftModel

    config = DFlashConfig(
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
    )
    model = DFlashDraftModel(config)
    for layer in model.layers:
        assert layer.attention_conv is None
        assert layer.mlp_conv is None


def test_conv_is_identity_at_init() -> None:
    """A freshly built DFlash2 conv must be a passthrough.

    The correction is learned on top of DFlash, so an untrained DFlash2 has to
    start numerically equal to DFlash rather than perturbing the backbone.
    """
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import torch

    from verl_speco.models.dflash2 import GroupedDynamicCausalConv

    conv = GroupedDynamicCausalConv(
        hidden_size=8, kernel_size=2, group_size=4, block_size=4
    )
    hidden = torch.randn(2, 8, 8)
    prepared, dynamic = conv.prepare(hidden)
    torch.testing.assert_close(prepared, hidden)
    torch.testing.assert_close(conv.finish(hidden, dynamic), hidden)


def test_conv_does_not_leak_across_block_boundaries() -> None:
    """Tap 1 is causal, so it must never read the previous block's last row.

    Training packs ``n_blocks`` blocks into one flat draft sequence. If the
    convolution ran over that flat axis, position 0 of block i would mix in the
    final position of block i-1, which belongs to an unrelated anchor.
    """
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import torch

    from verl_speco.models.dflash2 import GroupedDynamicCausalConv

    block_size = 4
    conv = GroupedDynamicCausalConv(
        hidden_size=8, kernel_size=2, group_size=4, block_size=block_size
    )
    # Make tap 1 the only contributor so any cross-block read is visible.
    with torch.no_grad():
        conv.base_kernel[0, 0, :] = 0.0
        conv.base_kernel[0, 1, :] = 1.0

    hidden = torch.randn(1, 2 * block_size, 8)
    out, _ = conv.prepare(hidden)

    # First row of each block has no in-block predecessor, so it must be zero.
    torch.testing.assert_close(out[:, 0], torch.zeros_like(out[:, 0]))
    torch.testing.assert_close(
        out[:, block_size], torch.zeros_like(out[:, block_size])
    )
    # Interior rows read their own block's previous row.
    torch.testing.assert_close(out[:, 1], hidden[:, 0])
    torch.testing.assert_close(out[:, block_size + 1], hidden[:, block_size])


def test_conv_rejects_length_that_is_not_a_block_multiple() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import torch

    from verl_speco.models.dflash2 import GroupedDynamicCausalConv

    conv = GroupedDynamicCausalConv(
        hidden_size=8, kernel_size=2, group_size=4, block_size=4
    )
    with pytest.raises(ValueError, match="multiple of block_size"):
        conv.prepare(torch.randn(1, 6, 8))


def test_selector_pair_scores_match_the_sequential_selector() -> None:
    """The vectorized training path must agree with the inference-time trace.

    ``pair_scores`` teacher-forces the predecessor; feeding it the path the
    sequential ``select`` actually took has to reproduce the same scores.
    """
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import torch

    from verl_speco.models.dflash2 import CandidateSelector

    torch.manual_seed(0)
    config = _tiny_dflash2_config()
    selector = CandidateSelector(config)

    batch, block = 2, config.block_size
    hidden = torch.randn(batch, block, config.hidden_size)
    logits = torch.randn(batch, block, config.vocab_size)
    anchor_ids = torch.randint(0, config.vocab_size, (batch,))

    path, candidates = selector.select(hidden, logits, anchor_ids)
    assert path.shape == (batch, block)

    # Rebuild the predecessor sequence the trace used: anchor, then its choices.
    predecessors = torch.cat([anchor_ids.unsqueeze(1), path[:, :-1]], dim=1)
    top_k = candidates.shape[-1]
    unary = torch.gather(logits, 2, candidates)
    with torch.no_grad():
        flat_scores = selector.pair_scores(
            hidden.reshape(-1, config.hidden_size),
            unary.reshape(-1, top_k),
            candidates.reshape(-1, top_k),
            predecessors.reshape(-1),
        ).reshape(batch, block, top_k)
    replayed = torch.gather(
        candidates, 2, flat_scores.argmax(dim=-1, keepdim=True)
    ).squeeze(-1)
    torch.testing.assert_close(replayed, path)


def test_dflash2_training_model_adds_selector_loss() -> None:
    """The selector objective must actually contribute to the total loss."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import torch

    from verl_speco.backends.dflash2_trainer_backend import DFlash2TrainingModel
    from verl_speco.models.dflash2 import DFlash2DraftModel

    config = _tiny_dflash2_config()
    bsz, seq_len = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (bsz, seq_len))
    hidden_states_list = [
        torch.randn(bsz, seq_len, config.target_hidden_size)
        for _ in config.target_layer_ids
    ]
    loss_mask = torch.ones(bsz, seq_len, dtype=torch.long)
    lm_head_weight = torch.randn(config.vocab_size, config.hidden_size)

    def run(selector_loss_weight):
        torch.manual_seed(0)
        model = DFlash2TrainingModel(
            draft_model=DFlash2DraftModel(config),
            block_size=config.block_size,
            num_anchors=config.num_anchors,
            selector_loss_weight=selector_loss_weight,
        )
        return model(input_ids, hidden_states_list, loss_mask, lm_head_weight)

    loss_off, _, _, _, _, diagnostics_off = run(0.0)
    loss_on, _, _, _, _, diagnostics_on = run(1.0)

    assert "selector_loss" not in diagnostics_off
    assert "selector_loss" in diagnostics_on
    assert float(diagnostics_on["selector_active_count"]) > 0
    # With the weight on, the total loss carries the extra selector term.
    assert float(loss_on) > float(loss_off)
    assert float(loss_on) == pytest.approx(
        float(loss_off) + float(diagnostics_on["selector_loss"]), rel=1e-4
    )


def test_selector_loss_does_not_backprop_into_the_backbone() -> None:
    """The selector must train as a re-ranking head, not reshape the drafter.

    torch.topk passes gradient through the selected logits and the selector's
    hidden projection reads the backbone state, so without detaching, the
    selector's cross-entropy would flow back into the draft model. That would
    both change the drafter's effective objective and make any DFlash/DFlash2
    comparison confounded, since the two backbones would no longer differ only
    by architecture.
    """
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import torch

    from verl_speco.backends.dflash2_trainer_backend import DFlash2TrainingModel
    from verl_speco.models.dflash2 import DFlash2DraftModel

    torch.manual_seed(0)
    config = _tiny_dflash2_config()
    model = DFlash2TrainingModel(
        draft_model=DFlash2DraftModel(config),
        block_size=config.block_size,
        num_anchors=config.num_anchors,
        selector_loss_weight=1.0,
    )

    n_rows = 12
    active_hidden = torch.randn(n_rows, config.hidden_size, requires_grad=True)
    active_logits = torch.randn(n_rows, config.vocab_size, requires_grad=True)
    bsz, n_blocks = 1, 3
    safe_label_indices = torch.arange(
        1, 1 + n_blocks * config.block_size
    ).reshape(bsz, n_blocks, config.block_size)
    active_mask = torch.zeros(
        bsz * n_blocks * config.block_size, dtype=torch.bool
    )
    active_mask[:n_rows] = True

    loss, _ = model._auxiliary_loss(
        input_ids=torch.randint(
            0, config.vocab_size, (bsz, n_blocks * config.block_size + 1)
        ),
        safe_label_indices=safe_label_indices,
        active_mask=active_mask,
        active_hidden=active_hidden,
        active_logits=active_logits,
        active_targets=torch.randint(0, config.vocab_size, (n_rows,)),
        active_weights=torch.ones(n_rows),
    )
    loss.backward()

    assert active_hidden.grad is None, (
        "selector loss leaked gradient into the backbone hidden states"
    )
    assert active_logits.grad is None, (
        "selector loss leaked gradient into the drafter logits via topk"
    )
    # The selector's own parameters must still be trained.
    assert model.draft_model.candidate_selector.hidden_projection.weight.grad is not None


def test_dflash2_training_model_rejects_restricted_vocab() -> None:
    """Selector candidates are real token ids, so a restricted vocab is invalid."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    from verl_speco.backends.dflash2_trainer_backend import DFlash2TrainingModel
    from verl_speco.models.dflash2 import DFlash2DraftModel

    config = _tiny_dflash2_config()
    with pytest.raises(ValueError, match="full_vocab"):
        DFlash2TrainingModel(
            draft_model=DFlash2DraftModel(config),
            block_size=config.block_size,
            num_anchors=config.num_anchors,
            loss_mode="restricted_ce",
        )


def test_backend_pins_conv_block_size_to_the_trainer_block_size() -> None:
    """The convs and the trainer must not disagree about the block size.

    The convs are built from the drafter config while the training wrapper takes
    its own ``dflash2_block_size``. If the training value is a multiple of the
    config value, ``_to_block_local``'s guard still passes but each conv "block"
    spans several anchor blocks, so the causal tap reads across an anchor
    boundary: exactly the leak the module is supposed to prevent, silently.
    """
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from omegaconf import OmegaConf

    from verl_speco.backends.dflash2_trainer_backend import DFlash2TrainerBackend
    from verl_speco.models.dflash2 import DFlash2Config

    backend = DFlash2TrainerBackend(
        OmegaConf.create(
            {
                "rollout": {
                    "drafter": {
                        "speculative_algorithm": "DFLASH2",
                        "model_path": "",
                        # Deliberately a multiple of the config's block_size=4,
                        # so the block-multiple guard alone would not catch it.
                        "training": {"dflash2_block_size": 8},
                    }
                },
                "model": {"path": ""},
            }
        ),
        None,
    )
    config = _tiny_dflash2_config(block_size=4)
    assert isinstance(config, DFlash2Config)
    assert backend._resolved_block_size(config) == 8


def test_dflash2_backend_is_registered_in_the_factory() -> None:
    from verl_speco.backends.factory import SUPPORTED_DRAFTER_ALGORITHMS

    assert "DFLASH2" in SUPPORTED_DRAFTER_ALGORITHMS


def test_dflash2_uses_dflash_aux_layers() -> None:
    from verl_speco.integration.oldlogprob_layer_ids import DFLASH_FAMILY_ALGORITHMS

    assert "DFLASH2" in DFLASH_FAMILY_ALGORITHMS


def test_dflash2_config_lifts_nested_dflash_config(tmp_path) -> None:
    """Upstream z-lab checkpoints nest the DFlash2 knobs under dflash_config."""
    pytest.importorskip("transformers")
    import json

    from verl_speco.models.dflash2 import DFlash2Config

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "architectures": ["DFlash2DraftModel"],
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "num_hidden_layers": 1,
                "vocab_size": 32,
                "dflash_config": {
                    "block_size": 8,
                    "conv_kernel_size": 2,
                    "conv_group_size": 16,
                    "selector_rank": 256,
                    "selector_top_k": 16,
                    "mask_token_id": 31,
                    "target_layer_ids": [1, 3],
                },
            }
        )
    )

    config = DFlash2Config.from_dflash2_pretrained(str(tmp_path))
    assert config.model_type == "dflash2"
    assert config.block_size == 8
    assert config.conv_kernel_size == 2
    assert config.conv_group_size == 16
    assert config.selector_rank == 256
    assert config.selector_top_k == 16
    assert config.mask_token_id == 31
    assert config.target_layer_ids == [1, 3]


def test_dflash2_config_routes_through_auto(tmp_path) -> None:
    pytest.importorskip("transformers")
    import json

    from verl_speco.models.auto import AutoDraftModelConfig
    from verl_speco.models.dflash2 import DFlash2Config

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "architectures": ["DFlash2DraftModel"],
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "num_hidden_layers": 1,
                "vocab_size": 32,
                "dflash_config": {"block_size": 8, "selector_top_k": 16},
            }
        ),
        encoding="utf-8",
    )

    loaded = AutoDraftModelConfig.from_file(str(config_path))
    assert isinstance(loaded, DFlash2Config)
    assert loaded.architectures == ["DFlash2DraftModel"]
    # The nested z-lab block must survive routing through AutoDraftModelConfig.
    assert loaded.block_size == 8
    assert loaded.selector_top_k == 16


def test_accepted_length_stops_at_the_first_mismatch() -> None:
    """Acceptance is a prefix property, not a sum of per-position marginals."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("safetensors")
    import torch

    from verl_speco.backends.dflash_trainer_backend import accepted_prefix_lengths

    scored = torch.ones(1, 4, 7, dtype=torch.bool)
    correct = torch.tensor(
        [
            [
                [1, 1, 1, 1, 1, 1, 1],  # every position correct
                [1, 1, 0, 1, 1, 1, 1],  # miss at 3 truncates the tail
                [0, 1, 1, 1, 1, 1, 1],  # miss at 1 accepts nothing
                [0, 0, 0, 0, 0, 0, 0],  # nothing correct
            ]
        ],
        dtype=torch.bool,
    )

    accepted = accepted_prefix_lengths(correct, scored)
    assert accepted.tolist() == [[7, 2, 0, 0]]
    # The marginals would have credited the truncated blocks with far more.
    assert correct.sum(dim=-1).tolist() == [[7, 6, 6, 0]]


def test_accepted_length_treats_unscored_positions_as_terminal() -> None:
    """A block that runs past its supervised region cannot keep accepting."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("safetensors")
    import torch

    from verl_speco.backends.dflash_trainer_backend import accepted_prefix_lengths

    correct = torch.ones(1, 1, 7, dtype=torch.bool)
    scored = torch.tensor([[[1, 1, 1, 0, 0, 0, 0]]], dtype=torch.bool)

    assert accepted_prefix_lengths(correct, scored).tolist() == [[3]]
