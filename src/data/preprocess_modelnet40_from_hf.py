#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
preprocess_modelnet40_from_hf.py

Generate train.npz / val.npz / test.npz:

- full_points:    [N, n_full, 3]    
- partial_points: [N, n_partial, 3] 
- labels:         [N]               
- class_names:    [40]              
"""

import argparse
import io
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm


def parse_txt_to_xyz(txt: str):
    lines = txt.splitlines()
    numeric_lines = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        if any(c.isalpha() for c in ln):
            continue

        tokens = ln.split(",")
        ok = True
        for t in tokens:
            t = t.strip()
            if not t:
                continue
            try:
                float(t)
            except ValueError:
                ok = False
                break
        if not ok:
            continue

        numeric_lines.append(ln)

    if len(numeric_lines) == 0:
        return None

    buf = io.StringIO("\n".join(numeric_lines))
    arr = np.loadtxt(buf, delimiter=",", dtype=np.float32)

    if arr.ndim == 1:
        arr = arr[None, :]

    if arr.shape[1] < 3:
        return None

    xyz = arr[:, :3].astype(np.float32)
    return xyz


def normalize_unit_sphere(xyz: np.ndarray) -> np.ndarray:
    centroid = xyz.mean(axis=0, keepdims=True)
    xyz = xyz - centroid
    norms = np.linalg.norm(xyz, axis=1)
    max_r = norms.max()
    if max_r > 0:
        xyz = xyz / max_r
    return xyz


def sample_points(xyz: np.ndarray, n_points: int) -> np.ndarray:
    n = xyz.shape[0]
    if n == 0:
        raise ValueError("sample_points: empty point cloud.")
    if n >= n_points:
        idx = np.random.choice(n, n_points, replace=False)
    else:
        base_idx = np.arange(n)
        pad_idx = np.random.choice(n, n_points - n, replace=True)
        idx = np.concatenate([base_idx, pad_idx], axis=0)
    return xyz[idx, :]


def make_partial(full_xyz: np.ndarray, n_partial: int) -> np.ndarray:
    n = full_xyz.shape[0]
    if n == 0:
        raise ValueError("make_partial: empty full_xyz.")

    normal = np.random.randn(3).astype(np.float32)
    norm = np.linalg.norm(normal)
    if norm < 1e-6:
        normal = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        normal /= norm

    dots = full_xyz @ normal  # [n]
    thresh = np.median(dots)
    mask = dots > thresh
    visible = full_xyz[mask]

    if visible.shape[0] < n_partial:
        return sample_points(full_xyz, n_partial)
    else:
        return sample_points(visible, n_partial)


def _parse_name_list(txt: str):
    names = []
    for line in txt.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        token = line.split()[0]          # e.g. 'airplane_0001' or 'airplane/airplane_0001.txt'

        token = token.split("/")[-1]     # e.g. 'airplane_0001.txt' or 'airplane_0001'

        if token.endswith(".txt"):
            token = token[:-4]           # 'airplane_0001'

        names.append(token)

    return names


def _parse_class_names_txt(txt: str):
    names = []
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        names.append(line)
    return names


def load_official_splits_and_classes(ds):
    train_txt = None
    test_txt = None
    shape_names_txt = None

    for ex in ds:
        key = ex["__key__"]
        if key == "./modelnet40_train":
            train_txt = ex["txt"]
        elif key == "./modelnet40_test":
            test_txt = ex["txt"]
        elif key == "./modelnet40_shape_names":
            shape_names_txt = ex["txt"]

        if train_txt is not None and test_txt is not None and shape_names_txt is not None:
            break

    if train_txt is None or test_txt is None:
        raise RuntimeError("Cannot find ./modelnet40_train or ./modelnet40_test")

    train_names_list = _parse_name_list(train_txt)
    test_names_list = _parse_name_list(test_txt)

    if shape_names_txt is not None:
        class_names = _parse_class_names_txt(shape_names_txt)
    else:
        all_cls = sorted(
            {name.split("/")[0] for name in (train_names_list + test_names_list)}
        )
        class_names = list(all_cls)

    class_to_idx = {c: i for i, c in enumerate(class_names)}

    print(f"[INFO] Official train number: {len(train_names_list)}, test number: {len(test_names_list)}")
    print(f"[INFO] Official class number: {len(class_names)}, eg: {class_names[:5]} ...")

    return train_names_list, test_names_list, class_names, class_to_idx


def process_with_official_split(
    ds,
    train_set,
    test_set,
    class_to_idx,
    class_names,
    n_full: int,
    n_partial: int,
    max_train: int | None,
    max_test: int | None,
    val_ratio: float,
    out_dir: Path,
):

    train_full_all = []
    train_partial_all = []
    train_label_all = []
    train_basename_all = []

    test_full_list = []
    test_partial_list = []
    test_label_list = []
    test_basename_list = []

    seen_train_names = set()
    seen_test_names = set()

    train_found_raw = 0
    test_found_raw = 0

    for ex in tqdm(ds, desc="Processing all samples with official split"):
        key = ex["__key__"]       # eg './airplane/airplane_0001'
        name = key[2:] if key.startswith("./") else key  # 'airplane/airplane_0001'

        parts = name.split("/")
        basename = parts[-1]      # 'airplane_0001'
        if basename.endswith(".txt"):
            basename = basename[:-4]

        in_train = basename in train_set
        in_test = basename in test_set

        if not (in_train or in_test):
            continue

        xyz = parse_txt_to_xyz(ex["txt"])
        if xyz is None:
            print(f"[WARN] Skip {name}")
            continue

        xyz = normalize_unit_sphere(xyz)
        full = sample_points(xyz, n_full)
        partial = make_partial(full, n_partial)

        cls_name = name.split("/")[0]
        if cls_name not in class_to_idx:
            print(f"[WARN] Class: {cls_name} not in class_to_idx, skip {name}")
            continue
        label = class_to_idx[cls_name]

        if in_train:
            if (max_train is not None) and (train_found_raw >= max_train):
                continue
            train_full_all.append(full.astype("float32"))
            train_partial_all.append(partial.astype("float32"))
            train_label_all.append(label)
            train_basename_all.append(basename)
            train_found_raw += 1
            seen_train_names.add(basename)
        elif in_test:
            if (max_test is not None) and (test_found_raw >= max_test):
                continue
            test_full_list.append(full.astype("float32"))
            test_partial_list.append(partial.astype("float32"))
            test_label_list.append(label)
            test_basename_list.append(basename)
            test_found_raw += 1
            seen_test_names.add(basename)

    if len(train_full_all) == 0:
        raise RuntimeError("train empty")
    if len(test_full_list) == 0:
        raise RuntimeError("test empty")

    num_train_total = len(train_full_all)
    num_val = int(num_train_total * val_ratio)
    num_val = max(1, min(num_train_total - 1, num_val))

    idx_perm = np.random.permutation(num_train_total)
    val_idx = idx_perm[:num_val]
    train_idx = idx_perm[num_val:]

    train_full_arr_all = np.stack(train_full_all, axis=0)
    train_partial_arr_all = np.stack(train_partial_all, axis=0)
    train_labels_arr_all = np.array(train_label_all, dtype=np.int64)

    train_full_arr = train_full_arr_all[train_idx]
    train_partial_arr = train_partial_arr_all[train_idx]
    train_labels_arr = train_labels_arr_all[train_idx]

    val_full_arr = train_full_arr_all[val_idx]
    val_partial_arr = train_partial_arr_all[val_idx]
    val_labels_arr = train_labels_arr_all[val_idx]

    test_full_arr = np.stack(test_full_list, axis=0)
    test_partial_arr = np.stack(test_partial_list, axis=0)
    test_labels_arr = np.array(test_label_list, dtype=np.int64)

    class_arr = np.array(class_names)

    out_train = out_dir / "train.npz"
    np.savez_compressed(
        out_train,
        full_points=train_full_arr,
        partial_points=train_partial_arr,
        labels=train_labels_arr,
        class_names=class_arr,
    )
    print(
        f"[INFO] Save train to {out_train}, num samples={train_full_arr.shape[0]} "
        f"(Official train number={len(train_set)}, actual number={len(seen_train_names)})"
    )

    out_val = out_dir / "val.npz"
    np.savez_compressed(
        out_val,
        full_points=val_full_arr,
        partial_points=val_partial_arr,
        labels=val_labels_arr,
        class_names=class_arr,
    )
    print(
        f"[INFO] Save val to {out_val}, num samples={val_full_arr.shape[0]} "
        f"(From train ratio: {val_ratio:.2f})"
    )

    out_test = out_dir / "test.npz"
    np.savez_compressed(
        out_test,
        full_points=test_full_arr,
        partial_points=test_partial_arr,
        labels=test_labels_arr,
        class_names=class_arr,
    )
    print(
        f"[INFO] Save test to {out_test}, num samples={test_full_arr.shape[0]} "
        f"(Official test num={len(test_set)}, actual number={len(seen_test_names)})"
    )


# -----------------------------
#  main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess ModelNet40 (official split) from Hugging Face."
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="output dir for train.npz / val.npz / test.npz",
    )
    parser.add_argument(
        "--n_full",
        type=int,
        default=2048,
        help="full sample number (default: 2048)",
    )
    parser.add_argument(
        "--n_partial",
        type=int,
        default=1024,
        help="partial sample number (default: 1024)",
    )
    parser.add_argument(
        "--max_train",
        type=int,
        default=None,
        help="debugging, None for all 9843。",
    )
    parser.add_argument(
        "--max_test",
        type=int,
        default=None,
        help="debugging, None for all 2468。",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="validation ratio from the train dataset (default: 0.2 for 20%%)。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed",
    )

    args = parser.parse_args()

    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading Hugging Face dataset: Pointcept/modelnet40_normal_resampled-compressed")
    ds = load_dataset("Pointcept/modelnet40_normal_resampled-compressed", split="train")
    print(f"[INFO] HF total number: {len(ds)}")

    train_names_list, test_names_list, class_names, class_to_idx = \
        load_official_splits_and_classes(ds)

    if args.max_train is not None:
        train_names_eff = train_names_list[: args.max_train]
    else:
        train_names_eff = train_names_list

    if args.max_test is not None:
        test_names_eff = test_names_list[: args.max_test]
    else:
        test_names_eff = test_names_list

    train_set = set(train_names_eff)
    test_set = set(test_names_eff)

    print(
        f"[INFO] Use official split: train={len(train_set)},test={len(test_set)},"
        f"class number={len(class_names)};val_ratio={args.val_ratio}"
    )

    process_with_official_split(
        ds=ds,
        train_set=train_set,
        test_set=test_set,
        class_to_idx=class_to_idx,
        class_names=class_names,
        n_full=args.n_full,
        n_partial=args.n_partial,
        max_train=args.max_train,
        max_test=args.max_test,
        val_ratio=args.val_ratio,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()
