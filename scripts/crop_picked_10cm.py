#!/usr/bin/env python3
"""Create fixed 10 cm crops and paired label artifacts for picked data."""

import argparse
import json
import shutil
import tarfile
from pathlib import Path

from PIL import Image, ImageDraw
from tqdm import tqdm

from src.config import CropRegion
from src.cropped_retrieval import process_cropped_folder
from src.label_processor import get_color_for_label, get_polygons_from_label, transform_label_json
from src.picked_10cm import CROP_BOUNDS, crop_image_to_canvas, crop_label_tar
from src.utils import find_image_files, get_label_tar_path


DEFAULT_INPUT_DIR = Path("/mnt/c/Users/zhangyutang/Desktop/picked")
DEFAULT_OUTPUT_DIR = Path("/mnt/c/Users/zhangyutang/Desktop/picked_10cm_cropped")
CROP_REGION = CropRegion(*CROP_BOUNDS)
VEIN_LABEL_IDS = {26, 27, 28, 29, 30, 31, 32}
ARTERY_LABEL_IDS = {3, 33, 34, 35, 36, 37, 38, 39, 40}
VEIN_COLOR = (0, 188, 212)
ARTERY_COLOR = (255, 82, 0)


def _read_label_json(tar_path: Path) -> dict:
    with tarfile.open(tar_path, "r:*") as archive:
        member = next(
            (
                item
                for item in archive.getmembers()
                if item.isfile() and item.name.lower().endswith(".json")
            ),
            None,
        )
        if member is None:
            raise ValueError(f"Label TAR does not contain JSON: {tar_path}")
        stream = archive.extractfile(member)
        if stream is None:
            raise ValueError(f"Unable to read label JSON: {tar_path}")
        return json.loads(stream.read().decode("utf-8"))


def _nifti_suffix(tar_path: Path) -> str | None:
    with tarfile.open(tar_path, "r:*") as archive:
        for member in archive.getmembers():
            name = member.name.lower()
            if member.isfile() and name.endswith(".nii.gz"):
                return ".nii.gz"
            if member.isfile() and name.endswith(".nii"):
                return ".nii"
    return None


def draw_label_outlines(
    image: Image.Image,
    label_data: dict,
    line_width: int = 2,
    label_ids: set[int] | None = None,
) -> Image.Image:
    """Draw source-color polygon outlines over an RGB image."""
    canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)

    for polygon in get_polygons_from_label(label_data):
        if label_ids is not None and polygon["label"] not in label_ids:
            continue
        points = [(point[0], point[1]) for point in polygon["points"] if len(point) >= 2]
        if not points:
            continue

        color = get_color_for_label(polygon["label"], label_data)
        if len(points) == 1:
            x, y = points[0]
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), outline="black", width=3)
            draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=color)
            continue

        path = points + [points[0]] if polygon.get("closed", False) else points
        draw.line(path, fill="black", width=line_width + 2, joint="curve")
        draw.line(path, fill=color, width=line_width, joint="curve")

    return canvas


def draw_vessel_outlines(image: Image.Image, label_data: dict, line_width: int = 2) -> Image.Image:
    """Draw only vessel boundaries with the fixed artery and vein palette."""
    canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)

    for polygon in get_polygons_from_label(label_data):
        label_id = polygon["label"]
        if label_id in ARTERY_LABEL_IDS:
            color = ARTERY_COLOR
        elif label_id in VEIN_LABEL_IDS:
            color = VEIN_COLOR
        else:
            continue

        points = [(point[0], point[1]) for point in polygon["points"] if len(point) >= 2]
        if not points:
            continue
        if len(points) == 1:
            x, y = points[0]
            draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=color)
            continue

        path = points + [points[0]] if polygon.get("closed", False) else points
        draw.line(path, fill=color, width=line_width, joint="curve")

    return canvas


def _save_cropped_image(image: Image.Image, path: Path) -> None:
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        image.save(path, quality=100, subsampling=0)
    else:
        image.save(path)


def _preflight(input_dir: Path) -> list[tuple[Path, Path]]:
    image_paths = find_image_files(input_dir)
    if not image_paths:
        raise ValueError(f"No supported images found in: {input_dir}")

    items = []
    for image_path in image_paths:
        label_path = get_label_tar_path(image_path)
        if label_path is None:
            raise ValueError(f"Missing label TAR for: {image_path.name}")
        with Image.open(image_path) as image:
            if image.size != (1920, 1080):
                raise ValueError(f"Expected 1920x1080 image, got {image.size}: {image_path.name}")
        _read_label_json(label_path)
        items.append((image_path, label_path))

    return items


def _process_item(image_path: Path, label_path: Path, output_root: Path) -> bool:
    frame_dir = output_root / image_path.stem
    frame_dir.mkdir()

    original_image_path = frame_dir / image_path.name
    original_label_path = frame_dir / label_path.name
    cropped_name = f"{image_path.stem}_cropped{image_path.suffix.lower()}"
    cropped_image_path = frame_dir / cropped_name
    cropped_label_path = frame_dir / f"{Path(cropped_name).stem}_{image_path.suffix.lstrip('.').lower()}_Label.tar"
    original_overlay_path = frame_dir / f"{image_path.stem}_original_overlay.png"
    cropped_overlay_path = frame_dir / f"{image_path.stem}_cropped_overlay.png"
    cropped_white_label_path = frame_dir / f"{image_path.stem}_cropped_label_white.png"
    cropped_vessel_overlay_path = frame_dir / f"{image_path.stem}_cropped_vessel_overlay.png"
    cropped_vessel_white_path = frame_dir / f"{image_path.stem}_cropped_vessel_label_white.png"

    shutil.copy2(image_path, original_image_path)
    shutil.copy2(label_path, original_label_path)
    label_data = _read_label_json(label_path)
    transformed_label = transform_label_json(label_data, CROP_REGION)

    with Image.open(image_path) as source_image:
        original_overlay = draw_label_outlines(source_image, label_data)
        cropped_image = crop_image_to_canvas(source_image)

    _save_cropped_image(cropped_image, cropped_image_path)
    original_overlay.save(original_overlay_path)
    draw_label_outlines(cropped_image, transformed_label).save(cropped_overlay_path)
    white_labels = Image.new("RGB", cropped_image.size, "white")
    draw_label_outlines(white_labels, transformed_label).save(cropped_white_label_path)
    draw_vessel_outlines(cropped_image, transformed_label).save(cropped_vessel_overlay_path)
    white_vessels = Image.new("RGB", cropped_image.size, "white")
    draw_vessel_outlines(white_vessels, transformed_label).save(cropped_vessel_white_path)

    suffix = _nifti_suffix(label_path)
    cropped_label_stem = Path(cropped_name).stem
    crop_label_tar(
        label_path,
        cropped_label_path,
        cropped_jpg_filename=cropped_name,
        json_member_basename=f"{cropped_label_stem}_{image_path.suffix.lstrip('.').lower()}_Label.json",
        nifti_member_basename=(
            f"{cropped_label_stem}_{image_path.suffix.lstrip('.').lower()}_Label{suffix}"
            if suffix is not None
            else None
        ),
    )
    retrieval = process_cropped_folder(frame_dir)
    return bool(retrieval.record["features"])


def run_batch(input_dir: Path = DEFAULT_INPUT_DIR, output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, int]:
    """Generate all per-frame crop artifacts without overwriting an existing result."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")

    items = _preflight(input_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = output_dir.with_name(f".{output_dir.name}.in_progress")
    if staging_dir.exists():
        raise FileExistsError(f"Staging directory already exists: {staging_dir}")

    gallery_records = 0
    try:
        staging_dir.mkdir()
        for image_path, label_path in tqdm(items, desc="Processing frames"):
            gallery_records += int(_process_item(image_path, label_path, staging_dir))
        staging_dir.replace(output_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    return {
        "total": len(items),
        "processed": len(items),
        "gallery_records": gallery_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Crop picked data to a fixed 10 cm square")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    result = run_batch(args.input_dir, args.output_dir)
    print(f"Processed {result['processed']} of {result['total']} frames")
    print(f"Retrieval gallery frames: {result['gallery_records']}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
