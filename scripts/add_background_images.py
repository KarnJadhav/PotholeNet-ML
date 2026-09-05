#!/usr/bin/env python3
"""
add_background_images.py — inject negative (no-pothole) images into an
ALREADY-PREPARED dataset, without redoing the full prepare_dataset.py merge.

Use this when your dataset was assembled from a pre-organized dump (not run
through prepare_dataset.py from raw sources) and you just need to top it up
with background examples: clean roads, shadows, cracks, manholes, puddles —
anything that should NOT fire a pothole detection.

Each background image gets an EMPTY label file (YOLO convention for
"no objects in this image" — validate_dataset.py already treats this as a
legitimate background image, just flags it as a warning for you to confirm).

Splits the new images across train/val/test using the SAME ratio as your
current dataset (measured from what's already there), grouped by filename-
prefix sequence heuristic so a burst of related negative shots doesn't leak
across splits, same logic as prepare_dataset.py.

Usage:
  python add_background_images.py --source /path/to/negative/images --dataset ../datasets/potholes
  python add_background_images.py --source /path/to/negative/images --dataset ../datasets/potholes --dry-run
"""
import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

SPLITS = ["train", "val", "test"]
VALID_EXTS = {".jpg", ".jpeg", ".png"}


def current_split_ratio(dataset_root: Path):
    counts = {}
    for split in SPLITS:
        img_dir = dataset_root / "images" / split
        counts[split] = len([p for p in img_dir.iterdir() if p.suffix.lower() in VALID_EXTS]) if img_dir.exists() else 0
    total = sum(counts.values())
    if total == 0:
        return {"train": 0.70, "val": 0.20, "test": 0.10}
    return {k: v / total for k, v in counts.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, type=Path, help="dir of new background/negative images")
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-fraction-of-positives", type=float, default=0.20,
                     help="cap: don't add more negatives than this fraction of your "
                          "CURRENT total positive images, to avoid tanking recall. "
                          "0.20 = negatives capped at 20%% of current dataset size.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    new_images = [p for p in args.source.iterdir() if p.suffix.lower() in VALID_EXTS]
    if not new_images:
        print(f"No images found in {args.source}")
        return

    current_total = sum(
        len([p for p in (args.dataset / "images" / s).iterdir() if p.suffix.lower() in VALID_EXTS])
        for s in SPLITS if (args.dataset / "images" / s).exists()
    )
    cap = int(current_total * args.max_fraction_of_positives)
    if len(new_images) > cap:
        print(f"WARNING: {len(new_images)} negative images found, but cap is {cap} "
              f"({args.max_fraction_of_positives*100:.0f}% of current {current_total} images). "
              f"Randomly sampling {cap} to avoid overloading the dataset with negatives.")
        random.Random(args.seed).shuffle(new_images)
        new_images = new_images[:cap]

    ratio = current_split_ratio(args.dataset)
    print(f"Current split ratio: {ratio}")

    # group by filename-prefix sequence heuristic (same as prepare_dataset.py default)
    groups = defaultdict(list)
    for img_path in new_images:
        seq_id = img_path.stem.rsplit("_", 1)[0]
        groups[f"bg_{seq_id}"].append(img_path)

    seq_ids = list(groups.keys())
    random.Random(args.seed).shuffle(seq_ids)

    target_train = len(new_images) * ratio["train"]
    target_val = len(new_images) * ratio["val"]

    assignment = {}
    running_train, running_val = 0, 0
    for seq_id in seq_ids:
        n = len(groups[seq_id])
        if running_train < target_train:
            assignment[seq_id] = "train"
            running_train += n
        elif running_val < target_val:
            assignment[seq_id] = "val"
            running_val += n
        else:
            assignment[seq_id] = "test"

    per_split_count = defaultdict(int)
    for seq_id, split in assignment.items():
        for img_path in groups[seq_id]:
            per_split_count[split] += 1
            dest_name = f"bg__{img_path.name}"
            img_dest = args.dataset / "images" / split / dest_name
            lbl_dest = args.dataset / "labels" / split / f"{Path(dest_name).stem}.txt"

            if args.dry_run:
                print(f"WOULD ADD [{split}] {img_path.name} -> {dest_name} (empty label)")
            else:
                shutil.copy2(img_path, img_dest)
                lbl_dest.write_text("")  # empty label file = background image, YOLO convention

    print(f"\n{'DRY RUN — nothing written' if args.dry_run else 'Added'}:")
    for split in SPLITS:
        print(f"  {split}: +{per_split_count[split]} background images")

    print(f"\nNext: re-run validate_dataset.py — these will show as 'empty annotation "
          f"file (background image?)' WARNINGS, which is correct and expected here. "
          f"Then re-run visualize_annotations.py to confirm none of them actually contain "
          f"a pothole you missed labeling.")


if __name__ == "__main__":
    main()
