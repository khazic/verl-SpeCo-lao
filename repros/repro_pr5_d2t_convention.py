"""Before/after repro for the d2t offset-convention default.

Runs unchanged on both `main` and the fix branch. It builds the EAGLE-3 and
P-EAGLE drafts with no frequency mapping supplied, then applies the exact formula
the serving engine uses to turn a drafted id back into a target id:

    targets = arange(draft_vocab_size) + d2t          # vLLM llama_eagle3.py

  main       -> d2t is arange, so the engine resolves draft id i to target id 2*i
                and the top of the range falls outside the target vocabulary
  fix branch -> d2t is zeros, the engine resolves draft id i to target id i
"""

from __future__ import annotations

import torch
from transformers import LlamaConfig

from verl_speco.data.preprocessing import process_token_dict_to_mappings
from verl_speco.models.eagle.llama_eagle import LlamaForCausalLMEagle3
from verl_speco.models.peagle import LlamaForCausalLMPeagle, PeagleConfig

VOCAB = 32


def eagle3_draft(draft_vocab_size: int) -> LlamaForCausalLMEagle3:
    return LlamaForCausalLMEagle3(
        LlamaConfig(
            hidden_size=8,
            intermediate_size=16,
            num_attention_heads=2,
            num_key_value_heads=2,
            num_hidden_layers=1,
            vocab_size=VOCAB,
            draft_vocab_size=draft_vocab_size,
            target_hidden_size=8,
            num_aux_hidden_states=3,
            pad_token_id=0,
            rms_norm_eps=1e-6,
            max_position_embeddings=64,
        )
    )


def peagle_draft(draft_vocab_size: int) -> LlamaForCausalLMPeagle:
    return LlamaForCausalLMPeagle(
        PeagleConfig(
            hidden_size=8,
            intermediate_size=16,
            num_attention_heads=2,
            num_key_value_heads=2,
            num_hidden_layers=1,
            num_draft_layers=1,
            target_hidden_size=8,
            num_aux_hidden_states=3,
            vocab_size=VOCAB,
            draft_vocab_size=draft_vocab_size,
            mask_token_id=VOCAB - 1,
            pad_token_id=0,
            rms_norm_eps=1e-6,
            max_position_embeddings=64,
        )
    )


def report(label: str, model) -> bool:
    d2t = model.d2t
    targets = torch.arange(model.draft_vocab_size) + d2t
    identity = bool(torch.equal(targets, torch.arange(model.draft_vocab_size)))
    in_range = bool(int(targets.max()) < model.vocab_size)
    print(f"          {label}")
    print(f"            d2t[:8]              = {d2t[:8].tolist()}")
    print(f"            engine target ids[:8]= {targets[:8].tolist()}")
    print(f"            max target id        = {int(targets.max())} (vocab_size={model.vocab_size})")
    print(f"            identity mapping     = {identity}")
    print(f"            all ids in range     = {in_range}")
    return identity and in_range


def main() -> None:
    print("[what the repo's own mapping generator emits]")
    from collections import Counter

    d2t, _ = process_token_dict_to_mappings(
        Counter({3: 10, 9: 8, 17: 5, 21: 2}),
        draft_vocab_size=4,
        target_vocab_size=VOCAB,
    )
    print(f"          used_tokens          = [3, 9, 17, 21]")
    print(f"          generated d2t        = {d2t.tolist()}")
    print(f"          arange(4) + d2t      = {(torch.arange(4) + d2t).tolist()}")
    print("          so d2t holds OFFSETS, and the identity mapping is all zeros")
    print()

    print("[full-vocabulary draft, no mapping supplied]")
    ok_eagle3 = report("EAGLE-3", eagle3_draft(VOCAB))
    ok_peagle = report("P-EAGLE", peagle_draft(VOCAB))
    print()

    print("[verdict]")
    if ok_eagle3 and ok_peagle:
        print("          RESULT: the default d2t is the offset identity (fixed)")
    else:
        print("          RESULT: the default d2t resolves to the wrong target ids (bug present)")


if __name__ == "__main__":
    main()
