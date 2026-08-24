"""Before/after repro for the SGLang hidden-state request gate.

Runs unchanged on both `main` and the fix branch. For every algorithm that
`speco_worker` can build a training backend for, it asks two questions:

  1. what hidden-state layout does the drafter's training loop need? This comes
     from `resolve_drafter_hidden_states_layout`, which is the shared resolver
     `speco_worker` and `speco_ray_trainer` already use.
  2. what does the SGLang rollout actually request for it?

  main       -> EAGLE1, EAGLE2, PEAGLE and DOMINO request nothing, so drafter
                training on an SGLang rollout gets no hidden states at all
  fix branch -> every algorithm requests the layout its trainer needs
"""

from __future__ import annotations

from verl_speco.integration.oldlogprob_layer_ids import (
    resolve_drafter_hidden_states_layout,
)
from verl_speco.integration.sglang_runtime import (
    _drafter_uses_dflash_aux_hidden,
    _drafter_uses_eagle_last_hidden,
)

# The algorithms speco_worker.build_trainer_backend accepts.
ALGORITHMS = ["EAGLE1", "EAGLE2", "EAGLE3", "PEAGLE", "DFLASH", "DSPARK", "DOMINO"]


def drafter_cfg(algorithm: str) -> dict:
    return {
        "enable": True,
        "enable_drafter_training": True,
        "speculative_algorithm": algorithm,
        "training": {
            "collect_hidden_states_from_sgl": True,
            "use_logits": False,
        },
    }


def main() -> None:
    header = f"{'algorithm':<10} {'trainer needs':<22} {'sglang requests':<22} verdict"
    print(header)
    print("-" * len(header))

    broken = []
    for algorithm in ALGORITHMS:
        config = drafter_cfg(algorithm)
        layout = resolve_drafter_hidden_states_layout(algorithm, config["training"])
        needs_last = layout.startswith("eagle3_")
        needs_aux = layout.startswith("dflash_")

        requests_last = _drafter_uses_eagle_last_hidden(config)
        requests_aux = _drafter_uses_dflash_aux_hidden(config)

        if requests_last:
            requested = "last_hidden"
        elif requests_aux:
            requested = "dflash_aux_hidden"
        else:
            requested = "NOTHING"

        ok = (needs_last and requests_last) or (needs_aux and requests_aux)
        if not ok:
            broken.append(algorithm)
        print(
            f"{algorithm:<10} {layout:<22} {requested:<22} {'ok' if ok else 'MISMATCH'}"
        )

    print()
    print("[verdict]")
    if broken:
        print(f"          algorithms with no hidden states: {', '.join(broken)}")
        print("          RESULT: drafter training is starved on SGLang (bug present)")
    else:
        print("          every algorithm requests the layout its trainer needs")
        print("          RESULT: gate matches the shared resolver (fixed)")


if __name__ == "__main__":
    main()
