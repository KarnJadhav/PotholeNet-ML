#!/usr/bin/env python3
"""
filter_attain_negatives.py — pull pothole-free images out of the Attain
dataset, split into PRIORITY (faded marking / patch — targets the confirmed
false-positive bug) and GENERIC (any other pothole-free image) sets.

Handles all three Attain sub-repo formats:
  - Attain_SMP_OS_V1.0: YOLO segmentation polygons, only 2 classes
    (Alligator crack, Linear crack) — no pothole class exists in this
    sub-repo at all, so every image qualifies automatically. Polygon
    content is never parsed since we're discarding original labels anyway
    (our detector only cares about "pothole present: yes/no").
  - Attain_SMP_WS_V1.0: YOLO bbox, 16 classes (ids 11,12 = Pothole).
  - Attain_SMP_WS_V2.0: Pascal VOC XML, free-text class names.

Output: two folders of images (no labels — add_background_images.py
creates the empty label files), plus a manifest CSV for provenance/
attribution tracking (matters for Attain's CC BY 4.0 requirement).

Usage:
  python filter_attain_negatives.py --attain-root /mnt/d/Pot_Dataset/Attain --out attain_negatives
"""
import argparse
import csv
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

# WS_V1.0 class-index -> name, from its data.yaml (order matters, 0-indexed)
WS_V1_CLASSES = [
    "Alligator crack - High", "Alligator crack - Low", "Alligator crack - low",
    "Block crack - Low", "Faded marking - High", "Faded marking - Low",
    "Linear crack - High", "Linear crack - Low", "Manhole - High", "Manhole - Low",
    "Patch - Low", "Pothole - High", "Pothole - Low", "Raveling - Low",
    "Weathering - High", "Weathering - Low",
]
WS_V1_POTHOLE_IDS = {WS_V1_CLASSES.index("Pothole - High"), WS_V1_CLASSES.index("Pothole - Low")}
WS_V1_PRIORITY_IDS = {
    WS_V1_CLASSES.index("Faded marking - High"), WS_V1_CLASSES.index("Faded marking - Low"),
    WS_V1_CLASSES.index("Patch - Low"),
}

V2_POTHOLE_NAMES = {"Pothole - High", "Pothole - Low"}
V2_PRIORITY_NAMES = {
    "Faded marking - High", "Faded marking - Low",
    "Patch and utility cut- High", "Patch and utility cut- Low",  # note: no space before dash, matches source data
}

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def scan_os_v1(root: Path):
    """No pothole class exists in this sub-repo — every image qualifies as generic negative."""
    img_dir = root / "Images"
    results = []
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() in IMG_EXTS:
            results.append((img_path, "generic", "no pothole class in this sub-repo"))
    print(f"[OS_V1.0] {len(results)} images, all qualify (no pothole class exists)")
    return results


def scan_ws_v1(root: Path):
    img_dir, lbl_dir = root / "Images", root / "Labels"
    results = []
    skipped_pothole = 0
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue
        class_ids = set()
        for line in lbl_path.read_text().strip().splitlines():
            parts = line.split()
            if parts:
                class_ids.add(int(parts[0]))

        if class_ids & WS_V1_POTHOLE_IDS:
            skipped_pothole += 1
            continue

        if class_ids & WS_V1_PRIORITY_IDS:
            present = [WS_V1_CLASSES[i] for i in class_ids]
            results.append((img_path, "priority", "; ".join(present)))
        else:
            present = [WS_V1_CLASSES[i] for i in class_ids] if class_ids else ["(empty)"]
            results.append((img_path, "generic", "; ".join(present)))

    print(f"[WS_V1.0] {len(results)} qualify, {skipped_pothole} excluded (contain real pothole)")
    return results


def scan_ws_v2(root: Path):
    img_dir, lbl_dir = root / "Images", root / "Labels"
    results = []
    skipped_pothole = 0
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        xml_path = lbl_dir / f"{img_path.stem}.xml"
        if not xml_path.exists():
            continue

        names = set()
        try:
            tree = ET.parse(xml_path)
            for obj in tree.getroot().findall("object"):
                name_el = obj.find("name")
                if name_el is not None and name_el.text:
                    names.add(name_el.text.strip())
        except ET.ParseError:
            continue

        if names & V2_POTHOLE_NAMES:
            skipped_pothole += 1
            continue

        if names & V2_PRIORITY_NAMES:
            results.append((img_path, "priority", "; ".join(sorted(names))))
        else:
            results.append((img_path, "generic", "; ".join(sorted(names)) or "(empty)"))

    print(f"[WS_V2.0] {len(results)} qualify, {skipped_pothole} excluded (contain real pothole)")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attain-root", required=True, type=Path,
                     help="path to the Attain folder containing the 3 sub-repos")
    ap.add_argument("--out", type=Path, default=Path("attain_negatives"))
    args = ap.parse_args()

    all_results = []
    os_v1 = args.attain_root / "Attain_SMP_OS_V1.0"
    ws_v1 = args.attain_root / "Attain_SMP_WS_V1.0"
    ws_v2 = args.attain_root / "Attain_SMP_WS_V2.0"

    if os_v1.exists():
        all_results += scan_os_v1(os_v1)
    if ws_v1.exists():
        all_results += scan_ws_v1(ws_v1)
    if ws_v2.exists():
        all_results += scan_ws_v2(ws_v2)

    priority_dir = args.out / "priority"
    generic_dir = args.out / "generic"
    priority_dir.mkdir(parents=True, exist_ok=True)
    generic_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.out / "manifest.csv"
    n_priority, n_generic = 0, 0
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "category", "source_classes_present", "source_subrepo"])
        for img_path, category, classes_present in all_results:
            dest_dir = priority_dir if category == "priority" else generic_dir
            dest_name = f"attain_{img_path.name}"
            shutil.copy2(img_path, dest_dir / dest_name)
            writer.writerow([dest_name, category, classes_present, img_path.parent.parent.name])
            if category == "priority":
                n_priority += 1
            else:
                n_generic += 1

    print(f"\nTotal qualifying images: {n_priority + n_generic}")
    print(f"  priority (faded marking / patch — targets the confirmed bug): {n_priority}")
    print(f"  generic (other pothole-free): {n_generic}")
    print(f"\nManifest (for CC BY 4.0 attribution/provenance tracking): {manifest_path}")
    print(f"\nNext: python add_background_images.py --source {priority_dir} --dataset ../datasets/potholes")
    print("Run priority/ first — it's small and directly targets the confirmed failure. "
          "Only add generic/ afterward if you want more general negative coverage; "
          "add_background_images.py's 20% cap re-checks your CURRENT dataset size each time, "
          "so running it twice sequentially is safe.")


if __name__ == "__main__":
    main()
