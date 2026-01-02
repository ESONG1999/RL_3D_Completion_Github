#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
download dataset
"""

import importlib
import sys
from pathlib import Path
import io
import numpy as np


def ensure_lib(pkg_name: str, pip_name: str | None = None):
    pip_name = pip_name or pkg_name
    try:
        importlib.import_module(pkg_name)
    except ImportError:
        print(f"[ERROR] Python package `{pkg_name}` not installed.")
        print(f"        please run:  pip install {pip_name}")
        sys.exit(1)


def main():
    ensure_lib("datasets")
    ensure_lib("numpy")

    from datasets import load_dataset

    print("[INFO] Loading Pointcept/modelnet40_normal_resampled-compressed ...")
    ds = load_dataset(
        "Pointcept/modelnet40_normal_resampled-compressed",
        split="train",
    )
    print(ds)
    print(f"[INFO] sample number: {len(ds)}")

    def parse_txt(txt: str) -> np.ndarray:
        buf = io.StringIO(txt)
        arr = np.loadtxt(buf, delimiter=",", dtype=np.float32)  # [N, 6]
        return arr

    ex = ds[0]
    print("[INFO] key:", ex["__key__"])
    pts = parse_txt(ex["txt"])
    print("[INFO] points shape:", pts.shape)  # (2048, 6)

    out_dir = Path("data/processed_from_hf")
    out_dir.mkdir(parents=True, exist_ok=True)

    full_list = []
    label_list = []
    key_list = []

    for i in range(min(100, len(ds))):
        ex = ds[i]
        arr = parse_txt(ex["txt"])
        xyz = arr[:, :3]
        full_list.append(xyz.astype("float32"))
        key_list.append(ex["__key__"])
        cls = ex["__key__"].split("/")[1]
        label_list.append(cls)

    np.savez_compressed(
        out_dir / "modelnet40_example_100.npz",
        points=np.stack(full_list, axis=0),
        labels=np.array(label_list),
        keys=np.array(key_list),
    )
    print(f"[INFO] Save to {out_dir / 'modelnet40_example_100.npz'}")


if __name__ == "__main__":
    main()
