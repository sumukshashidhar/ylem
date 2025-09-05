#!/usr/bin/env python3
"""Very small smoke test for ylem.Cleaner.

Uses a tiny Hugging Face model for a quick run.
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    # Use a tiny model to keep it fast and lightweight.
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    # Ensure nano alias uses Gemma 3 270M IT.
    os.environ.setdefault("YLEM_MODEL_NANO", "google/gemma-3-270m-it")

    import ylem

    if len(sys.argv) <= 1:
        print("Usage: python scripts/smoke_cleaner.py 'your input text'", file=sys.stderr)
        return 2
    text = sys.argv[1]

    cleaner = ylem.Cleaner("nano")
    out = cleaner(text, max_new_tokens=32, do_sample=False)
    print("INPUT:\n", text)
    print("\nOUTPUT:\n", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
