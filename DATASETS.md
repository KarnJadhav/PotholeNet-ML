# Dataset Sources & License Compliance

This file tracks every external dataset used or considered for PotholeNet-ML,
its license status, and what's still unverified. **This is not legal advice**
— it's a compliance tracker to keep unresolved items visible instead of
silently assumed. Verify anything marked ⚠️ or ❌ directly at the source
before commercial deployment, redistribution of trained weights, or public
release of the dataset itself.

Status legend: ✅ confirmed from source · ⚠️ partially confirmed / verify
before relying on it · ❌ not yet checked

---

## Currently used in `datasets/potholes/`

### RDD2022 (Road Damage Dataset)
- **Source:** figshare, https://doi.org/10.6084/m9.figshare.21431547
- **License status:** ⚠️ **Not confirmed.** The dataset's figshare page
  describes intended usage but does not display an explicit license badge
  in what's checkable from the page content. Third-party metadata
  aggregators list the dataset's *metadata* under CC0 1.0 — that is not
  the same thing as the image license.
- **Action required:** Visit the figshare page directly and check the
  "License" field in the sidebar before any commercial use or
  redistribution of images derived from this source.
- **Used for:** D40 (pothole) class extraction only, via `prepare_dataset.py`.

### Attain (pavement distress dataset)
- **Source:** Mendeley Data, https://data.mendeley.com/datasets/nykrzdm74f/1
  (companion paper: Rezaeimanesh et al., *Data in Brief*, 2025,
  doi:10.1016/j.dib.2025.111715)
- **License status:** ✅ **CC BY 4.0** — stated explicitly on the companion
  paper. Data in Brief (Elsevier) policy requires the dataset to carry the
  same license as the article. Verify the Mendeley dataset page shows the
  same badge before relying on this for commercial use — reasonably
  confident but not independently viewed on the Mendeley page itself.
- **Attribution required under CC BY 4.0:** credit the original authors
  (Rezaeimanesh, Golroo, Fahmani, Amani, Hasanitabaar, Entezari, Karimi —
  Amirkabir University of Technology) and note changes made (filtered to
  faded-marking and patch/utility-cut classes only; used as negative,
  no-pothole training examples; a subset was further color/brightness
  augmented in one experiment).
- **Used for:** ✅ **Actively in use.** 317 images (faded-marking + patch
  classes, explicitly excluding any image also containing a
  `Pothole - High/Low` annotation) injected as negative training examples
  via `filter_attain_negatives.py` + `add_background_images.py`. Confirmed
  to measurably fix a false-positive on painted road markings; did not
  fully resolve a related false-positive on patch/repair textures (see
  `README.md`'s Known Limitation section). A further 152-image
  color-augmented derivative of this data was generated and briefly used
  in a training experiment that was **rejected and rolled back** — those
  derivative images remain in `datasets/potholes/` (filenames matching
  `bg__aug*`) but are not part of the currently deployed model's training
  data. The CC BY attribution above applies to this derivative work too.

---

## Considered but not integrated

Identified as candidates during dataset research; **none of these have
been downloaded or merged into `datasets/potholes/`.**

### Pothole-600
- **Source:** https://sites.google.com/view/pothole-600/dataset (also
  mirrored on Kaggle — search "Pothole-600" directly, exact listing not
  confirmed here)
- **Format mismatch, blocking integration:** provides RGB +
  transformed-disparity stereo image pairs with **pixel-level segmentation
  masks**, not YOLO bounding-box labels. `prepare_dataset.py`'s current
  `--pothole600-images`/`--pothole600-labels` flags assume plain YOLO
  `.txt` bbox format, which does not match this dataset's actual
  structure. A mask→bbox conversion step would need to be built before
  this data could be ingested at all.
- **License status:** ❌ **Never verified.** No explicit license terms
  found on the official page or in citing literature checked so far.
- **Status:** Not integrated, and format conversion is the blocking step
  — resolve that before the license check becomes relevant. Listed here
  as a documented future option, not an active dependency.

| Source                                                | Platform                 | License status                                                       |
|-------------------------------------------------------|--------------------------|----------------------------------------------------------------------|
| BharatPotHole                                         | Kaggle                   | ❌ Not checked                                                       |
| Indian Roads Dataset                                  | Kaggle                   | ❌ Not checked                                                       |
| road-pothole-images-for-pothole-detection (sovitrath) | Kaggle                   | ❌ Not checked                                                       |
| Bharat AI SoC Pothole Detector                        | Roboflow Universe        | ⚠️ Platform default CC BY 4.0 unless overridden — verify per-project |
| Public Roboflow Pothole dataset                       | public.roboflow.com      | ⚠️ Same as above                                                     |
| EMT (Emirates Multi-Task)                             | UAE dashcam              | ❌ Rejected                                                          |

**Before integrating any of these:** find the stated license (and, for
Pothole-600, build the mask→bbox conversion first), update this section
with a ✅ and confirmed terms — don't carry ❌/⚠️ into actual use.

---

## Trained model weights

Weights (`best.pt`, `best.onnx`, hosted at
`huggingface.co/Karn81/PotholeNet-YOLO11n`) are derived from **RDD2022
(⚠️ unconfirmed license) + Attain (✅ CC BY 4.0, confirmed) + custom
images.** Pothole-600 has never been integrated and is not part of this
derivation. **The weights' redistribution terms are governed by whichever
input dataset's license is most restrictive** — not by this project's own
MIT code license (see `LICENSE.md`). Until RDD2022 is resolved, treat the
trained model's licensing status as **unresolved**, regardless of
deployment status.

## Compliance checklist before public/commercial deployment

- [ ] RDD2022 license confirmed directly from figshare
- [x] Attain dataset license confirmed (CC BY 4.0) — double-check the
      Mendeley page directly still recommended before commercial use
- [x] Attain CC BY 4.0 attribution text drafted (above) — apply it in any
      public-facing model card or dataset release
- [ ] Pothole-600 — not currently a dependency; if integrated later,
      resolve the mask→bbox format conversion FIRST, then confirm license
      before use (not a pre-deployment blocker as things stand today)
- [ ] Any newly-added dataset has its license recorded here BEFORE being
      merged into `datasets/potholes/`
- [ ] If any source turns out non-commercial/share-alike, decide whether
      to exclude it, re-license accordingly, or seek explicit permission