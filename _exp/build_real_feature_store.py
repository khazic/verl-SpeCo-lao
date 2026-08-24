"""Build DFlash-family feature stores from a real conversation dataset.

Unlike ci/standalone_feature_store_smoke.py (8 hardcoded prompts), this reads
real multi-turn conversations from a parquet dataset and writes a train store
and a disjoint held-out store, so a drafter can be evaluated on sequences it
never trained on.

The loss mask covers assistant turns only, matching what an RL rollout supervises.
"""

from __future__ import annotations

import argparse

import pyarrow.parquet as pq
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from verl_speco.models.dflash.modeling_dflash import build_target_layer_ids
from verl_speco.trainer.feature_store import DraftFeatureSample, TorchShardFeatureStore


def _iter_conversations(parquet_path: str, limit: int):
    pf = pq.ParquetFile(parquet_path)
    seen = 0
    for batch in pf.iter_batches(batch_size=64, columns=["conversations"]):
        for row in batch.column("conversations").to_pylist():
            turns = [
                {"role": t["role"], "content": t["content"]}
                for t in row
                if t.get("role") in ("user", "assistant") and t.get("content")
            ]
            if len(turns) < 2 or turns[0]["role"] != "user":
                continue
            yield turns
            seen += 1
            if seen >= limit:
                return


def _encode(tokenizer, turns, max_len):
    """Tokenize a conversation and mark assistant tokens in the loss mask."""
    ids: list[int] = []
    mask: list[float] = []
    for turn in turns:
        rendered = tokenizer.apply_chat_template(
            [turn], tokenize=False, add_generation_prompt=False
        )
        piece = tokenizer(rendered, add_special_tokens=False)["input_ids"]
        ids.extend(piece)
        mask.extend([1.0 if turn["role"] == "assistant" else 0.0] * len(piece))
        if len(ids) >= max_len:
            break
    return ids[:max_len], mask[:max_len]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--out-train", required=True)
    parser.add_argument("--out-heldout", required=True)
    parser.add_argument("--num-train", type=int, default=768)
    parser.add_argument("--num-heldout", type=int, default=256)
    parser.add_argument("--max-len", type=int, default=384)
    parser.add_argument("--num-context-layers", type=int, default=5)
    parser.add_argument("--algorithm", default="DFLASH")
    args = parser.parse_args()

    device = "cuda"
    print(f"[features] loading target {args.target}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.target)
    target = (
        AutoModelForCausalLM.from_pretrained(args.target, torch_dtype=torch.bfloat16)
        .to(device)
        .eval()
    )
    target_cfg = AutoConfig.from_pretrained(args.target)
    num_layers = int(getattr(target_cfg, "num_hidden_layers"))
    target_layer_ids = build_target_layer_ids(args.num_context_layers, num_layers)
    print(f"[features] target_layer_ids={target_layer_ids} (of {num_layers})", flush=True)

    total = args.num_train + args.num_heldout
    stores = {
        "train": TorchShardFeatureStore(
            args.out_train,
            max_samples_per_shard=32,
            metadata={"algorithm": args.algorithm, "target_model_path": args.target},
            shard_prefix="train",
        ),
        "heldout": TorchShardFeatureStore(
            args.out_heldout,
            max_samples_per_shard=32,
            metadata={"algorithm": args.algorithm, "target_model_path": args.target},
            shard_prefix="heldout",
        ),
    }

    written = {"train": 0, "heldout": 0}
    for index, turns in enumerate(_iter_conversations(args.parquet, total * 3)):
        if written["train"] + written["heldout"] >= total:
            break
        ids, mask = _encode(tokenizer, turns, args.max_len)
        # Need enough supervised tokens for anchors to be sampleable.
        if len(ids) < 64 or sum(mask) < 16:
            continue
        split = "train" if written["train"] < args.num_train else "heldout"
        if split == "heldout" and written["heldout"] >= args.num_heldout:
            break

        input_ids = torch.tensor(ids, dtype=torch.long)
        with torch.no_grad():
            out = target(
                input_ids=input_ids.unsqueeze(0).to(device), output_hidden_states=True
            )
        # HF hidden_states[0] is the embedding output, so layer L lands at L + 1.
        blocks = [out.hidden_states[i + 1][0] for i in target_layer_ids]
        hidden_states = torch.cat(blocks, dim=-1).to(torch.bfloat16).cpu()

        seq_len = int(input_ids.numel())
        stores[split].write_many(
            [
                DraftFeatureSample(
                    algorithm=args.algorithm,
                    input_ids=input_ids,
                    loss_mask=torch.tensor(mask, dtype=torch.float32),
                    hidden_states=hidden_states,
                    position_ids=torch.arange(1, seq_len + 1, dtype=torch.long),
                    metadata={
                        "source": "perfectblend_real",
                        "global_step": 0,
                        "target_model_path": args.target,
                        "hidden_states_layout": "dflash_aux",
                        "target_layer_ids": target_layer_ids,
                        "use_logits": False,
                        "sequence_length": seq_len,
                        "loss_tokens": int(sum(mask)),
                        "full_sequence_length": seq_len,
                        "feature_start": 0,
                        "feature_end": seq_len,
                    },
                )
            ]
        )
        written[split] += 1
        if (written["train"] + written["heldout"]) % 100 == 0:
            print(f"[features] wrote {written}", flush=True)

    for store in stores.values():
        store.close()
    print(f"[features] DONE {written}", flush=True)


if __name__ == "__main__":
    main()
