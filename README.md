# EUS-cropper

Chinese: [README_zh-CN.md](README_zh-CN.md)

## Purpose

EUS-cropper is the fixed 10 cm EUS image-and-label crop stage of a larger research pipeline. It creates paired image, annotation, visualization, organ-metadata, and two-dimensional vessel-retrieval outputs from a directory of labelled EUS frames.

## What This Repository Does

The sole public workflow is:

```bash
python scripts/crop_picked_10cm.py --input-dir /path/to/input --output-dir /path/to/output
```

It crops each source frame to the fixed pixel bounds `(603, 123, 1563, 1083)`, transforms the paired label TAR, optionally crops its two-dimensional NIfTI label image, generates visualization artifacts, and extracts vessel-retrieval metadata.

## Repository Scope and Data Availability

This is a reviewer and collaborator reproducibility package. It contains source code, tests, and environment definitions only. It does not include medical images, labels, patient information, or generated crop results. Place private input data outside the repository or in a Git-ignored directory.

## Requirements

- Linux, macOS, or Windows Subsystem for Linux (WSL) with an Ubuntu-compatible shell.
- [Mamba](https://mamba.readthedocs.io/) or a compatible conda installation.
- An input directory satisfying the contract below.

Native Windows Python has not been tested. On Windows, use a WSL path such as `/mnt/c/Users/<user>/...` rather than a backslash-separated Windows path.

## Installation

Clone or unpack this repository, then run these commands from its root:

```bash
mamba env create --file environment.yml
mamba activate eus-cropper
```

The checked-in environment uses Python 3.12.10 and fixed versions of Pillow, NumPy, NiBabel, SciPy, tqdm, and pytest. `requirements.txt` provides the equivalent runtime-only pip requirements.

## Input Data Contract

`--input-dir` must be a flat directory. Subdirectories are not scanned. Every supported image must meet all requirements below:

- Image extension is `.png`, `.jpg`, or `.jpeg`.
- Image dimensions are exactly `1920 x 1080` pixels.
- A paired TAR is in the same directory.
- For `frame_00000001.jpg`, the expected TAR name is `frame_00000001_jpg_Label.tar`.
- For an image whose stem itself contains periods, the implementation also accepts the stem with periods replaced by underscores before the extension suffix.
- The TAR contains one label JSON member. It may additionally contain one two-dimensional `.nii` or `.nii.gz` label image.

The JSON carries the polygon boundaries used in overlays. If a NIfTI member is available, it is also cropped and used for connected-component retrieval features. All input image/TAR pairs are checked before output creation begins; a missing pair, unsupported image size, or unreadable JSON stops the batch.

## Run the Complete Fixed 10 cm Crop

Choose a new output path that does not already exist:

```bash
python scripts/crop_picked_10cm.py \
  --input-dir /path/to/input \
  --output-dir /path/to/eus_10cm_results
```

The process creates a sibling staging directory named `.<output-name>.in_progress` and replaces it with the requested output directory only after all frames succeed. Existing output and staging directories cause an error rather than being overwritten.

The cropped canvas is `960 x 960` pixels. Source rows `y=123` through `y=1079` fill its first 957 rows; its final three rows are white image padding and zero-valued NIfTI padding. JSON polygon coordinates are translated and clipped to the same fixed crop region.

## Output Layout

The output root contains one directory per input frame plus `eus_possible_organs.json`. Each frame directory contains 14 files:

| File pattern | Description |
| --- | --- |
| `<frame>.<ext>` | Byte-for-byte copy of the original image. |
| `<frame>_<ext>_Label.tar` | Byte-for-byte copy of the original label TAR. |
| `<frame>_cropped.<ext>` | Fixed 10 cm, `960 x 960` cropped image. |
| `<frame>_cropped_<ext>_Label.tar` | Cropped JSON label TAR and, when present, cropped NIfTI label image. |
| `<frame>_original_overlay.png` | Original image with all original label boundaries. |
| `<frame>_cropped_overlay.png` | Cropped image with all cropped label boundaries. |
| `<frame>_cropped_label_white.png` | White background with all cropped label boundaries. |
| `<frame>_cropped_vessel_overlay.png` | Cropped image with generic artery/vein boundaries. |
| `<frame>_cropped_vessel_label_white.png` | White background with generic artery/vein boundaries. |
| `<frame>_original_ivc_ao_pv_overlay.png` | Original image with aorta, IVC, and portal-system boundaries. |
| `<frame>_cropped_ivc_ao_pv_overlay.png` | Cropped image with aorta, IVC, and portal-system boundaries. |
| `<frame>_cropped_ivc_ao_pv_label_white.png` | White background with aorta, IVC, and portal-system boundaries. |
| `<frame>_cropped_retrieval_features.json` | Organ labels, generic vessel features, anatomical vessel features, and skipped components. |
| `<frame>_cropped_gallery.jsonl` | One compatibility record for a downstream two-dimensional vessel gallery. |

## Label Semantics and Vessel Colours

Generic vessel visualization groups EUS labels `26-32` as veins and `3, 33-40` as arteries:

- Vein: cyan-blue, `RGB (0, 188, 212)`.
- Artery: orange-red, `RGB (255, 82, 0)`.

The three anatomical-vessel visualizations use a separate, fixed palette:

| Anatomical group | EUS label IDs | Boundary colour |
| --- | --- | --- |
| Aorta (Ao) | `3, 33` | Red, `RGB (255, 0, 0)` |
| Inferior vena cava (IVC) | `30` | Blue, `RGB (0, 0, 255)` |
| Portal venous system (PV) | `26, 27, 28, 29` | Purple, `RGB (170, 85, 255)` |

The portal venous system includes portal vein and branches, portal confluence, superior mesenteric vein, and splenic vein. `eus_possible_organs.json` lists all possible EUS organ labels used by the batch output; aorta, IVC, and portal vein retain both organ and vessel roles.

## Retrieval Features and Crop-Boundary Rule

`cropped-retrieval-features/v2` records both generic artery/vein and anatomical Ao/IVC/PV features. For NIfTI labels, the pipeline uses eight-connected two-dimensional components. A component that touches any edge of the `960 x 960` crop is treated as crop-truncated:

- It is excluded from `features` or `anatomical_vessel_features`.
- It is retained with `reason: "touches_image_edge"` in the corresponding skipped-components list.
- Its matching JSON contour is still drawn in all applicable boundary visualizations.

This rule applies to feature eligibility only. It never removes visible polygon boundaries from the all-label, generic-vessel, or Ao/IVC/PV overlay outputs. If an input TAR has no NIfTI member, image/JSON crops and visualizations are still generated, while NIfTI-derived feature arrays contain no components.

All feature positions are synthetic two-dimensional crop-plane coordinates: the origin is the cropped image's top-left corner, `x` increases to the right, `y` increases downward, and the nominal extent is `100 mm x 100 mm`. They are not patient-space three-dimensional positions.

## Validation

After activating the environment, run:

```bash
pytest -q
```

The test suite constructs synthetic image, TAR, JSON, and NIfTI fixtures. It verifies fixed crop bounds, white padding, JSON clipping, NIfTI geometry preservation, input pairing, output artifacts, organ metadata, generic retrieval features, and anatomical Ao/IVC/PV feature behavior.

## Limitations

- The public workflow accepts only flat, `1920 x 1080` input directories.
- The crop coordinates are fixed and are not configurable at runtime.
- The code processes two-dimensional image and NIfTI labels only.
- Original medical data are not distributed with this package.
- Native Windows Python is outside the tested platform scope.

## Citation

Please cite the associated manuscript when using this reproducibility package. No repository DOI or author list is defined here because this code is distributed as one stage of that manuscript's workflow.
