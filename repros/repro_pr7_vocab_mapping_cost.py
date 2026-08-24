"""Before/after repro for the draft vocabulary mapping build cost.

Runs unchanged on both `main` and the fix branch. It times
`process_token_dict_to_mappings` at a moderate vocabulary size and extrapolates
to a production tokenizer.

  main       -> t2d is built with `i in used_tokens` for every target id, so the
                cost is target_vocab_size * draft_vocab_size list comparisons
  fix branch -> t2d is filled by a single scatter
"""

from __future__ import annotations

import io
import time
from collections import Counter
from contextlib import redirect_stdout

import torch

from verl_speco.data.preprocessing import process_token_dict_to_mappings

TARGET_VOCAB = 65536
DRAFT_VOCAB = 8192

# A realistic production pair, for the extrapolation only.
PROD_TARGET_VOCAB = 151936
PROD_DRAFT_VOCAB = 32768


def main() -> None:
    token_dict = Counter({token: token + 1 for token in range(DRAFT_VOCAB * 2)})

    started = time.perf_counter()
    with redirect_stdout(io.StringIO()):
        d2t, t2d = process_token_dict_to_mappings(
            token_dict, DRAFT_VOCAB, TARGET_VOCAB
        )
    elapsed = time.perf_counter() - started

    print("[mapping build]")
    print(f"          target_vocab_size = {TARGET_VOCAB}")
    print(f"          draft_vocab_size  = {DRAFT_VOCAB}")
    print(f"          elapsed           = {elapsed:.3f}s")
    print()

    print("[result is unchanged either way]")
    selected = torch.nonzero(t2d, as_tuple=False).flatten()
    print(f"          t2d selects       = {int(t2d.sum())} tokens")
    print(f"          d2t decodes to    = {(torch.arange(DRAFT_VOCAB) + d2t).equal(selected)}")
    print()

    scale = (PROD_TARGET_VOCAB / TARGET_VOCAB) * (PROD_DRAFT_VOCAB / DRAFT_VOCAB)
    print("[extrapolated to a production tokenizer]")
    print(f"          target_vocab_size = {PROD_TARGET_VOCAB}")
    print(f"          draft_vocab_size  = {PROD_DRAFT_VOCAB}")
    print(f"          quadratic term is {scale:.0f}x the measured case")
    print(f"          projected         = {elapsed * scale:.1f}s if the cost is quadratic")
    print()

    print("[verdict]")
    if elapsed < 0.5:
        print("          RESULT: build is vectorized, cost does not scale with the product (fixed)")
    else:
        print("          RESULT: build scales with target_vocab_size * draft_vocab_size (bug present)")


if __name__ == "__main__":
    main()
