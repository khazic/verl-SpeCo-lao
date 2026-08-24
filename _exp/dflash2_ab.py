"""DFlash vs DFlash2 A/B on a real feature store, with held-out evaluation.

Both arms see identical data, identical seed, identical steps and lr; the only
difference is the backend. Reports held-out per-block-position accuracy, which
is the quantity block drafters are actually judged on, plus DFlash2's selector
lift over unary-only ranking.
"""

from __future__ import annotations

import argparse
import json

import torch
from omegaconf import OmegaConf

from verl_speco.trainer.feature_store import TorchShardFeatureStore


def _load(store_dir, limit=None):
    store = TorchShardFeatureStore(store_dir)
    keys = list(store.iter_keys())
    if limit:
        keys = keys[:limit]
    return store, keys


def _batch(store, key, device, num_context_layers):
    sample = store.read(key)
    hidden = sample.hidden_states
    if isinstance(hidden, list):
        hidden = torch.cat(hidden, dim=-1)
    return {
        "input_ids": sample.input_ids.unsqueeze(0).to(device),
        "loss_mask": sample.loss_mask.unsqueeze(0).to(device),
        "hidden_states": hidden.unsqueeze(0).to(device).to(torch.bfloat16),
        "attention_mask": torch.ones_like(sample.input_ids).unsqueeze(0).to(device),
    }


def _build(
    algorithm,
    target_path,
    target_cfg,
    block_size,
    num_context_layers,
    lr,
    selector_weight=None,
):
    prefix = algorithm.lower()
    training = {
        f"{prefix}_block_size": block_size,
        f"{prefix}_num_anchors": 64,
        f"{prefix}_num_target_layers": num_context_layers,
        f"{prefix}_num_hidden_layers": 1,
        "lr": lr,
    }
    if selector_weight is not None:
        training[f"{prefix}_selector_loss_weight"] = float(selector_weight)
    cfg = OmegaConf.create(
        {
            "rollout": {
                "drafter": {
                    "speculative_algorithm": algorithm,
                    "model_path": "/dev/null/does-not-exist",
                    "training": training,
                }
            },
            "model": {"path": target_path},
        }
    )
    from verl_speco.backends.factory import build_trainer_backend

    backend = build_trainer_backend(cfg, target_cfg)
    model, drafter_cfg = backend.build_model()
    device = "cuda"
    model = model.to(device).to(torch.bfloat16).train()
    backend.target_lm_head = backend.target_lm_head.to(device).to(torch.bfloat16)
    optimizer = backend.setup_optimizer(model, cfg.rollout.drafter.training)
    return backend, model, optimizer, drafter_cfg


@torch.no_grad()
def _evaluate(backend, model, store, keys, device, num_context_layers, block_size):
    """Held-out per-position accuracy, plus selector lift when present."""
    model.eval()
    correct = torch.zeros(block_size, dtype=torch.float64)
    counts = torch.zeros(block_size, dtype=torch.float64)
    sel_correct = sel_base = sel_tokens = 0.0
    for key in keys:
        batch = _batch(store, key, device, num_context_layers)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = backend.compute_loss(model, batch, 0)
        d = out["diagnostics"]
        correct += d["correct_per_position"].detach().double().cpu()
        counts += d["count_per_position"].detach().double().cpu()
        if "selector_correct_count" in d:
            sel_correct += float(d["selector_correct_count"])
            sel_base += float(d["selector_base_correct_count"])
            sel_tokens += float(d["selector_token_count"])
    model.train()
    per_pos = (correct / counts.clamp(min=1)).tolist()
    # Position 0 is the anchor slot and never carries loss weight.
    scored = counts[1:].sum().item()
    overall = (correct[1:].sum() / max(scored, 1)).item()
    result = {"overall_acc": overall, "per_position_acc": per_pos, "tokens": scored}
    if sel_tokens > 0:
        result["selector_acc"] = sel_correct / sel_tokens
        result["unary_only_acc"] = sel_base / sel_tokens
        result["selector_lift"] = (sel_correct - sel_base) / sel_tokens
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--train-store", required=True)
    parser.add_argument("--heldout-store", required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--num-context-layers", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-samples", type=int, default=128)
    parser.add_argument("--out", default="")
    parser.add_argument(
        "--arms",
        default="DFLASH,DFLASH2",
        help='Comma list; an entry may be "ALGO" or "ALGO:selector_weight"',
    )
    args = parser.parse_args()

    from transformers import AutoConfig

    target_cfg = AutoConfig.from_pretrained(args.target)
    device = "cuda"

    train_store, train_keys = _load(args.train_store)
    heldout_store, heldout_keys = _load(args.heldout_store, args.eval_samples)
    print(
        f"[ab] train_samples={len(train_keys)} heldout_samples={len(heldout_keys)}",
        flush=True,
    )

    # Arms are "ALGORITHM" or "ALGORITHM:selector_weight". The zero-weight
    # DFlash2 arm isolates the convolution: the selector head is then never
    # trained, so it also cannot inflate the global gradient-clipping norm and
    # shrink the backbone's effective step, which is the one indirect path by
    # which the selector could otherwise affect the drafter.
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    report = {}
    for arm in arms:
        algorithm, _, weight = arm.partition(":")
        selector_weight = float(weight) if weight else None
        torch.manual_seed(0)
        backend, model, optimizer, drafter_cfg = _build(
            algorithm,
            args.target,
            target_cfg,
            args.block_size,
            args.num_context_layers,
            args.lr,
            selector_weight=selector_weight,
        )
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[ab] {arm} trainable_params={n_params:,}", flush=True)

        history = []
        order = list(train_keys)
        for step in range(args.steps):
            key = order[step % len(order)]
            batch = _batch(train_store, key, device, args.num_context_layers)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = backend.compute_loss(model, batch, 0)
            loss = out["total_local_ploss"] / out["local_num_tokens"].clamp_min(1)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
                evaluation = _evaluate(
                    backend,
                    model,
                    heldout_store,
                    heldout_keys,
                    device,
                    args.num_context_layers,
                    args.block_size,
                )
                evaluation["step"] = step + 1
                evaluation["train_loss"] = float(loss)
                history.append(evaluation)
                extra = ""
                if "selector_lift" in evaluation:
                    extra = (
                        f"  selector_acc={evaluation['selector_acc']:.4f}"
                        f"  unary_only={evaluation['unary_only_acc']:.4f}"
                        f"  lift={evaluation['selector_lift']:+.4f}"
                    )
                print(
                    f"[ab] {arm} step {step + 1:4d}  train_loss={float(loss):.4f}"
                    f"  heldout_acc={evaluation['overall_acc']:.4f}{extra}",
                    flush=True,
                )
        report[arm] = history
        del backend, model, optimizer
        torch.cuda.empty_cache()

    final = {a: h[-1] for a, h in report.items()}
    print("\n[ab] ===== HELD-OUT SUMMARY =====", flush=True)
    for arm_name, evaluation in final.items():
        print(
            f"[ab] {arm_name:14s} overall={evaluation['overall_acc']:.4f}  "
            f"per_position={[round(v, 4) for v in evaluation['per_position_acc'][1:]]}",
            flush=True,
        )
    base = final.get("DFLASH")
    if base is not None:
        for arm_name, evaluation in final.items():
            if arm_name == "DFLASH":
                continue
            delta = evaluation["overall_acc"] - base["overall_acc"]
            print(f"[ab] {arm_name} - DFLASH delta = {delta:+.4f}", flush=True)
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(report, handle, indent=2)


if __name__ == "__main__":
    main()
