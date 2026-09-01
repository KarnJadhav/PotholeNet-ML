#!/usr/bin/env python3
"""
inspect_dedupe_clusters.py — run BEFORE dedupe_dataset.py's real deletion pass.

Shows cluster size distribution (large clusters = likely transitive chaining,
not real duplicates) and saves side-by-side montages of sample clusters so you
can visually confirm they're actually the same photo before anything gets deleted.

Usage:
  python inspect_dedupe_clusters.py --dataset ../datasets/potholes --phash-thresh 6 --sample 15
  python inspect_dedupe_clusters.py --dataset ../datasets/potholes --phash-thresh 3 --sample 15
"""
import argparse
import hashlib
import random
from collections import defaultdict
from pathlib import Path

import cv2
import imagehash
from PIL import Image

SPLITS = ["train", "val", "test"]
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


def build_montage(paths, out_path, tile=200):
    imgs = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = cv2.resize(img, (tile, tile))
        cv2.putText(img, p.parent.parent.name + "/" + p.name[:18], (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)
        imgs.append(img)
    if not imgs:
        return
    row = cv2.hconcat(imgs)
    cv2.imwrite(str(out_path), row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--phash-thresh", type=int, default=6)
    ap.add_argument("--sample", type=int, default=15, help="number of clusters to render for review")
    ap.add_argument("--out", type=Path, default=Path("dedupe_review"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    items = []
    for split in SPLITS:
        img_dir = args.dataset / "images" / split
        if img_dir.exists():
            items += [p for p in img_dir.iterdir() if p.suffix.lower() in VALID_EXTS]

    print(f"Scanning {len(items)} images...")
    md5s, phashes = {}, {}
    for p in items:
        img = cv2.imread(str(p))
        if img is None:
            continue
        md5s[p] = md5_of(p)
        try:
            phashes[p] = imagehash.phash(Image.open(p))
        except Exception:
            phashes[p] = None

    valid = [p for p in items if p in md5s]
    uf = UnionFind(valid)

    by_md5 = defaultdict(list)
    for p in valid:
        by_md5[md5s[p]].append(p)
    for group in by_md5.values():
        for p in group[1:]:
            uf.union(group[0], p)

    n = len(valid)
    for i in range(n):
        if phashes[valid[i]] is None:
            continue
        for j in range(i + 1, n):
            if phashes[valid[j]] is None:
                continue
            if phashes[valid[i]] - phashes[valid[j]] <= args.phash_thresh:
                uf.union(valid[i], valid[j])

    clusters = defaultdict(list)
    for p in valid:
        clusters[uf.find(p)].append(p)

    dup_clusters = {k: v for k, v in clusters.items() if len(v) > 1}
    sizes = sorted((len(v) for v in dup_clusters.values()), reverse=True)

    print(f"\n=== Cluster size distribution (phash_thresh={args.phash_thresh}) ===")
    print(f"Total duplicate clusters: {len(dup_clusters)}")
    print(f"Largest 20 cluster sizes: {sizes[:20]}")
    size_hist = defaultdict(int)
    for s in sizes:
        size_hist[s] += 1
    for size in sorted(size_hist):
        print(f"  size {size}: {size_hist[size]} cluster(s)")
    print(f"\nIf you see clusters of 5+ images, especially with sizes climbing into "
          f"double digits, that's very likely transitive chaining (A~B~C merged even "
          f"though A and C aren't really the same photo), not genuine duplication. "
          f"Real duplicates from 2-3 merged sources should mostly be size 2-3.")

    args.out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    sample_keys = random.sample(list(dup_clusters.keys()), min(args.sample, len(dup_clusters)))
    # also always include the largest clusters, most likely to be false chains
    largest_keys = sorted(dup_clusters, key=lambda k: -len(dup_clusters[k]))[:5]
    for i, key in enumerate(set(sample_keys) | set(largest_keys)):
        members = dup_clusters[key]
        out_path = args.out / f"cluster_{i:03d}_size{len(members)}.jpg"
        build_montage(members[:8], out_path)  # cap at 8 tiles wide for sanity

    print(f"\nWrote {len(set(sample_keys) | set(largest_keys))} cluster montages to {args.out}/")
    print("Open these and check: are the tiled images actually the same photo, or just "
          "visually similar road/pothole shots? If mostly the latter, raise --phash-thresh "
          "requirement (lower the number, e.g. 2 or 3) and re-run before deleting anything.")


if __name__ == "__main__":
    main()
