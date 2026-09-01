#!/usr/bin/env python3
"""
dedupe_dataset.py — fix an ALREADY-SPLIT dataset that has exact and/or
cross-split near-duplicate images (the leakage validate_dataset.py warns about).

Why this exists: prepare_dataset.py tags images by source/sequence but only
checks for filename-stem collisions, not pixel-content duplicates ACROSS
sources. If two source dumps contain the same underlying images under
different filenames (common with re-hosted Kaggle pothole sets), you get
silent duplicates that validate_dataset.py catches only after the split.

What it does:
  1. Hashes every image in train/val/test with md5 (exact) AND phash (near-dup).
  2. Union-find clusters images that are exact OR phash-close (<=--phash-thresh)
     into duplicate groups — transitively, so A~B~C group together even if
     A and C aren't directly close.
  3. For each group with >1 image, picks ONE split to keep the group in
     (priority: test > val > train, so the scarcest/most-important splits keep
     their representative and lose the fewest images) and ONE image+label pair
     within that split (first by filename), deleting every other copy
     (image + its label file) from disk.
  4. Also removes any image flagged corrupted/unreadable (cv2 can't decode it).
  5. Prints before/after counts per split.

This does NOT re-run the train/val/test ratio — it only removes redundant
copies, so your split ratios will shift slightly. Re-run validate_dataset.py
after this; if ratios drifted too far from 70/20/10, you may want to rebalance
by moving some intact groups between splits (this script doesn't do that).

Usage:
  python dedupe_dataset.py --dataset ../datasets/potholes --phash-thresh 6
  python dedupe_dataset.py --dataset ../datasets/potholes --dry-run   # preview only
"""
import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

import cv2
import imagehash
from PIL import Image, UnidentifiedImageError

SPLITS = ["train", "val", "test"]
SPLIT_PRIORITY = {"test": 0, "val": 1, "train": 2}  # lower = kept preferentially
VALID_EXTS = {".jpg", ".jpeg", ".png"}


class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_all_images(dataset_root: Path):
    """Returns list of (split, image_path)."""
    items = []
    for split in SPLITS:
        img_dir = dataset_root / "images" / split
        if not img_dir.exists():
            continue
        for p in sorted(img_dir.iterdir()):
            if p.suffix.lower() in VALID_EXTS:
                items.append((split, p))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--phash-thresh", type=int, default=6)
    ap.add_argument("--exact-only", action="store_true",
                     help="skip phash near-duplicate clustering entirely — only remove/consolidate "
                          "byte-identical (md5) images. Use this when phash is flagging visually "
                          "similar-but-genuinely-different photos (e.g. similar pavement/camera angle) "
                          "as false-positive duplicates.")
    ap.add_argument("--dry-run", action="store_true", help="report what would be removed, change nothing")
    args = ap.parse_args()

    items = collect_all_images(args.dataset)
    print(f"Scanning {len(items)} images across {SPLITS}...")

    md5s, phashes, corrupted = {}, {}, []
    for split, path in items:
        img = cv2.imread(str(path))
        if img is None:
            corrupted.append((split, path))
            continue
        md5s[path] = md5_of(path)
        if not args.exact_only:
            try:
                phashes[path] = imagehash.phash(Image.open(path))
            except UnidentifiedImageError:
                corrupted.append((split, path))

    valid_paths = [p for _, p in items if p in md5s]
    uf = UnionFind(valid_paths)

    # exact-duplicate clustering (md5) — always applied
    by_md5 = defaultdict(list)
    for p in valid_paths:
        by_md5[md5s[p]].append(p)
    for group in by_md5.values():
        for p in group[1:]:
            uf.union(group[0], p)

    if args.exact_only:
        print("Running in --exact-only mode: md5 exact matches only, no phash near-dup clustering.")
    else:
        # near-duplicate clustering (phash), O(n^2) — fine up to a few thousand images
        n = len(valid_paths)
        for i in range(n):
            for j in range(i + 1, n):
                pi, pj = valid_paths[i], valid_paths[j]
                if phashes[pi] - phashes[pj] <= args.phash_thresh:
                    uf.union(pi, pj)

    groups = defaultdict(list)
    path_to_split = {p: s for s, p in items}
    for p in valid_paths:
        groups[uf.find(p)].append(p)

    to_delete = []  # (split, image_path)
    dup_groups = 0
    for root, members in groups.items():
        if len(members) <= 1:
            continue
        dup_groups += 1
        # pick winning split by priority, then winning file by name within that split
        members_sorted = sorted(members, key=lambda p: (SPLIT_PRIORITY[path_to_split[p]], p.name))
        keeper = members_sorted[0]
        for loser in members_sorted[1:]:
            to_delete.append((path_to_split[loser], loser))

    to_delete += corrupted

    print(f"\nDuplicate clusters found: {dup_groups}")
    print(f"Corrupted images found:  {len(corrupted)}")
    print(f"Total images to remove:  {len(to_delete)}")

    before_counts = defaultdict(int)
    for split, _ in items:
        before_counts[split] += 1

    removed_counts = defaultdict(int)
    for split, img_path in to_delete:
        removed_counts[split] += 1
        lbl_path = args.dataset / "labels" / split / f"{img_path.stem}.txt"
        if args.dry_run:
            print(f"WOULD REMOVE [{split}] {img_path.name}  (+ label {lbl_path.name})")
        else:
            img_path.unlink(missing_ok=True)
            lbl_path.unlink(missing_ok=True)

    print(f"\n{'=== DRY RUN — nothing deleted ===' if args.dry_run else '=== Applied ==='}")
    for split in SPLITS:
        after = before_counts[split] - removed_counts[split]
        print(f"  {split}: {before_counts[split]} -> {after}  (-{removed_counts[split]})")

    if args.dry_run:
        print("\nRe-run without --dry-run to actually apply removals.")
    else:
        print("\nNow re-run validate_dataset.py — duplicate/leakage errors should be gone.")
        print("Check split ratios above; if drifted far from 70/20/10, consider re-splitting from raw sources instead.")


if __name__ == "__main__":
    main()
