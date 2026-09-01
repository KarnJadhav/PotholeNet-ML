#!/usr/bin/env python3
"""
prepare_dataset.py — merge RDD2022 (D40 class), Pothole-600, and your custom
Indian-road images into one YOLO-format dataset with sequence-aware splitting.

This does NOT download anything — point it at data you've already obtained and
checked the license for. It normalizes formats and builds the split.

--- Expected inputs ---

RDD2022 (Pascal VOC XML annotations, one XML per image):
  --rdd2022-images  dir of .jpg
  --rdd2022-annots  dir of .xml (same stem as image)
  Only <name>D40</name> boxes are kept; everything else is dropped.

Pothole-600 (already YOLO-labeled, or falls back to a simple CSV):
  --pothole600-images dir of .jpg/.png
  --pothole600-labels dir of .txt in YOLO format (class ignored, forced to 0)
  If your copy ships VOC/CSV instead, convert it to YOLO .txt first —
  this script assumes YOLO-format input for this source.

Custom Indian-road set (already labeled by you, YOLO format):
  --custom-images dir of images
  --custom-labels dir of .txt YOLO labels (same stem as image)

--- Sequence-aware splitting ---

To avoid leakage, images from the same source video/burst must land in the
SAME split. Provide a sequence id per image via one of:
  (a) filename prefix before the last underscore, e.g. "seq012_0043.jpg" -> "seq012"
      (default behavior)
  (b) a manifest CSV: --sequence-manifest path.csv with columns "filename,sequence_id"

Splitting is done by GROUP (sequence_id), not by individual image, targeting
70/20/10 train/val/test by image count (approximate, since groups vary in size).

Usage:
  python prepare_dataset.py \\
    --rdd2022-images raw/rdd2022/images --rdd2022-annots raw/rdd2022/annotations \\
    --pothole600-images raw/pothole600/images --pothole600-labels raw/pothole600/labels \\
    --custom-images raw/custom/images --custom-labels raw/custom/labels \\
    --out datasets/potholes --seed 42
"""
import argparse
import csv
import hashlib
import random
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

TARGET_CLASS = "D40"  # RDD2022 pothole class
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def voc_to_yolo_boxes(xml_path: Path):
    """Parse a Pascal VOC XML, return YOLO lines for TARGET_CLASS boxes only."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    img_w = float(size.find("width").text)
    img_h = float(size.find("height").text)

    lines = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        if name != TARGET_CLASS:
            continue
        bnd = obj.find("bndbox")
        xmin = float(bnd.find("xmin").text)
        ymin = float(bnd.find("ymin").text)
        xmax = float(bnd.find("xmax").text)
        ymax = float(bnd.find("ymax").text)

        xc = ((xmin + xmax) / 2) / img_w
        yc = ((ymin + ymax) / 2) / img_h
        w = (xmax - xmin) / img_w
        h = (ymax - ymin) / img_h
        lines.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return lines


def collect_rdd2022(images_dir: Path, annots_dir: Path, source_tag: str):
    """Returns list of (image_path, yolo_lines, sequence_id)."""
    items = []
    skipped_no_d40 = 0
    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        xml_path = annots_dir / f"{img_path.stem}.xml"
        if not xml_path.exists():
            continue
        lines = voc_to_yolo_boxes(xml_path)
        if not lines:
            skipped_no_d40 += 1
            continue  # image has no D40 (pothole) instance — drop rather than keep as empty background
        seq_id = f"{source_tag}_{img_path.stem.rsplit('_', 1)[0]}"
        items.append((img_path, lines, seq_id))
    print(f"[RDD2022] kept {len(items)} images with D40 boxes, "
          f"skipped {skipped_no_d40} with no pothole instance")
    return items


def collect_yolo_source(images_dir: Path, labels_dir: Path, source_tag: str, sequence_manifest: dict | None):
    items = []
    missing_labels = 0
    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        lbl_path = labels_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            missing_labels += 1
            continue
        raw_lines = lbl_path.read_text().strip().splitlines()
        # force class id to 0 regardless of source labeling scheme
        lines = []
        for line in raw_lines:
            parts = line.split()
            if len(parts) != 5:
                continue
            lines.append(" ".join(["0"] + parts[1:]))

        if sequence_manifest and img_path.name in sequence_manifest:
            seq_id = f"{source_tag}_{sequence_manifest[img_path.name]}"
        else:
            seq_id = f"{source_tag}_{img_path.stem.rsplit('_', 1)[0]}"
        items.append((img_path, lines, seq_id))
    print(f"[{source_tag}] kept {len(items)} images, skipped {missing_labels} with no label file")
    return items


def load_sequence_manifest(path: Path | None):
    if not path:
        return None
    mapping = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            mapping[row["filename"]] = row["sequence_id"]
    return mapping


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class _UnionFind:
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


def dedupe_by_content(items, phash_thresh=6):
    """Drop items whose image content is a duplicate (exact md5 OR near-identical
    phash, e.g. same photo re-saved as .png vs .jpg from two re-hosted sources)
    of an EARLIER item. Keeps the first occurrence per cluster.
    O(n^2) phash comparison — fine for a few thousand images; if this gets slow
    on a much larger merge, restrict comparisons to same-sequence-prefix buckets."""
    import imagehash
    from PIL import Image

    paths = [img_path for img_path, _, _ in items]
    by_path = {img_path: (lines, seq_id) for img_path, lines, seq_id in items}

    md5s, phashes = {}, {}
    for p in paths:
        md5s[p] = md5_of(p)
        try:
            phashes[p] = imagehash.phash(Image.open(p))
        except Exception:
            phashes[p] = None

    uf = _UnionFind(paths)
    by_md5 = defaultdict(list)
    for p in paths:
        by_md5[md5s[p]].append(p)
    for group in by_md5.values():
        for p in group[1:]:
            uf.union(group[0], p)

    n = len(paths)
    for i in range(n):
        if phashes[paths[i]] is None:
            continue
        for j in range(i + 1, n):
            if phashes[paths[j]] is None:
                continue
            if phashes[paths[i]] - phashes[paths[j]] <= phash_thresh:
                uf.union(paths[i], paths[j])

    clusters = defaultdict(list)
    for p in paths:
        clusters[uf.find(p)].append(p)

    kept_paths = set()
    dropped = 0
    for members in clusters.values():
        members_sorted = sorted(members, key=lambda p: p.name)
        kept_paths.add(members_sorted[0])
        dropped += len(members_sorted) - 1

    kept = [(p, by_path[p][0], by_path[p][1]) for p in paths if p in kept_paths]
    if dropped:
        print(f"[dedupe] dropped {dropped} duplicate images (exact or near-identical "
              f"content across sources/formats)")
    return kept


def split_by_sequence(items, train_frac, val_frac, seed):
    """Group items by sequence_id, shuffle groups, assign whole groups to splits."""
    groups = defaultdict(list)
    for item in items:
        groups[item[2]].append(item)

    seq_ids = list(groups.keys())
    random.Random(seed).shuffle(seq_ids)

    total_imgs = len(items)
    target_train = total_imgs * train_frac
    target_val = total_imgs * val_frac

    split_map = {}
    running_train, running_val = 0, 0
    for seq_id in seq_ids:
        n = len(groups[seq_id])
        if running_train < target_train:
            split_map[seq_id] = "train"
            running_train += n
        elif running_val < target_val:
            split_map[seq_id] = "val"
            running_val += n
        else:
            split_map[seq_id] = "test"

    result = defaultdict(list)
    for seq_id, group_items in groups.items():
        result[split_map[seq_id]].extend(group_items)
    return result


def write_split(out_root: Path, split: str, items):
    img_out = out_root / "images" / split
    lbl_out = out_root / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    for img_path, lines, seq_id in items:
        dest_name = f"{seq_id}__{img_path.name}"  # prefix keeps provenance + avoids stem collisions across sources
        dest_stem = Path(dest_name).stem
        shutil.copy2(img_path, img_out / dest_name)
        (lbl_out / f"{dest_stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rdd2022-images", type=Path)
    ap.add_argument("--rdd2022-annots", type=Path)
    ap.add_argument("--pothole600-images", type=Path)
    ap.add_argument("--pothole600-labels", type=Path)
    ap.add_argument("--custom-images", type=Path)
    ap.add_argument("--custom-labels", type=Path)
    ap.add_argument("--sequence-manifest", type=Path, default=None,
                     help="optional CSV with columns filename,sequence_id to override the "
                          "default filename-prefix heuristic for grouping")
    ap.add_argument("--out", type=Path, default=Path("datasets/potholes"))
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--val-frac", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    manifest = load_sequence_manifest(args.sequence_manifest)
    all_items = []

    if args.rdd2022_images and args.rdd2022_annots:
        all_items += collect_rdd2022(args.rdd2022_images, args.rdd2022_annots, "rdd2022")
    if args.pothole600_images and args.pothole600_labels:
        all_items += collect_yolo_source(args.pothole600_images, args.pothole600_labels,
                                          "pothole600", manifest)
    if args.custom_images and args.custom_labels:
        all_items += collect_yolo_source(args.custom_images, args.custom_labels,
                                          "custom", manifest)

    if not all_items:
        print("No source directories provided — nothing to do. Pass at least one --*-images/--*-labels pair.")
        return

    print(f"\nTotal merged images before split: {len(all_items)}")

    all_items = dedupe_by_content(all_items)
    print(f"Total images after content dedup: {len(all_items)}")

    splits = split_by_sequence(all_items, args.train_frac, args.val_frac, args.seed)
    for split_name in ("train", "val", "test"):
        write_split(args.out, split_name, splits[split_name])
        print(f"  {split_name}: {len(splits[split_name])} images "
              f"across {len({i[2] for i in splits[split_name]})} sequences")

    print(f"\nWrote merged dataset to {args.out}")
    print("Next: python scripts/validate_dataset.py --dataset "
          f"{args.out}  (fix every ERROR before training)")


if __name__ == "__main__":
    main()
