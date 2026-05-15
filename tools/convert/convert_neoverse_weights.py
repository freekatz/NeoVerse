#!/usr/bin/env python3
import argparse
import os
import sys

import torch

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from neoverse.loaders.legacy_converter import save_split_legacy_neoverse


def parse_dtype(name):
    if name in {None, "", "none", "None"}:
        return None
    return getattr(torch, name)


def main():
    parser = argparse.ArgumentParser(description="Split legacy NeoVerse diffusion weights into the new neoverse package format.")
    parser.add_argument("input", help="Legacy NeoVerse diffusion checkpoint path.")
    parser.add_argument("output_dir", help="Directory for converted weights.")
    parser.add_argument("--torch_dtype", default=None, help="Optional torch dtype name, e.g. bfloat16.")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    parts = save_split_legacy_neoverse(
        args.input,
        args.output_dir,
        torch_dtype=parse_dtype(args.torch_dtype),
        device=args.device,
    )
    print(f"wrote {args.output_dir}")
    print(f"wan_dit tensors={len(parts['wan_dit'])}")
    print(f"control_branch tensors={len(parts['control_branch'])}")
    print(f"control_config={parts['control_config']}")


if __name__ == "__main__":
    main()
