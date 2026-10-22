# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
from __future__ import annotations

import pytest

from verl_speco.integration.opd_cotrain import validate_sync_opd_cotrain_config


def _config(*, opd=True, train=True, strategy="sync"):
    return {
        "distillation": {"enabled": opd},
        "actor_rollout_ref": {
            "actor": {"strategy": "fsdp2"},
            "rollout": {
                "drafter": {
                    "enable": True,
                    "enable_drafter_training": train,
                    "training": {
                        "mode": "online",
                        "collect_hidden_states_from_old_logprob": True,
                        "scheduler": {"execution": {"strategy": strategy}},
                    },
                },
            },
        },
    }


@pytest.mark.parametrize("strategy", ["rollout_idle_worker", "standalone_async"])
def test_opd_cotrain_rejects_non_sync_execution(strategy):
    with pytest.raises(ValueError, match=r"scheduler\.execution\.strategy=sync"):
        validate_sync_opd_cotrain_config(_config(strategy=strategy))


def test_opd_cotrain_accepts_sync_and_default_execution():
    config = _config()
    validate_sync_opd_cotrain_config(config)
    del config["actor_rollout_ref"]["rollout"]["drafter"]["training"]["scheduler"]
    validate_sync_opd_cotrain_config(config)


@pytest.mark.parametrize("opd,train", [(False, True), (True, False), (False, False)])
def test_non_cotrain_modes_leave_execution_strategy_unchanged(opd, train):
    validate_sync_opd_cotrain_config(
        _config(opd=opd, train=train, strategy="rollout_idle_worker")
    )
