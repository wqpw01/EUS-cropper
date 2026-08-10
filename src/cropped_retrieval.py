"""Extract retrieval features from one fixed 10 cm cropped label TAR."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tarfile
import tempfile
from typing import Any

import nibabel as nib
import numpy as np
from PIL import Image
from scipy import ndimage


VEIN_LABEL_IDS = frozenset({26, 27, 28, 29, 30, 31, 32})
ARTERY_LABEL_IDS = frozenset({3, 33, 34, 35, 36, 37, 38, 39, 40})
_SCHEMA_VERSION = "cropped-retrieval-features/v1"
ORGAN_LABEL_BY_ID = {
    1: "liver",
    2: "liver",
    3: "aorta",
    4: "pancreas",
    5: "pancreas",
    6: "spleen",
    7: "pancreas",
    8: "liver",
    9: "pancreas",
    10: "duodenum",
    11: "pancreas",
    14: "liver",
    18: "spleen",
    19: "pancreas",
    20: "pancreas",
    21: "duodenum",
    22: "adrenal_gland_left",
    23: "adrenal_gland_right",
    24: "kidney_left",
    25: "kidney_right",
    26: "portal_vein",
    27: "portal_vein",
    30: "inferior_vena_cava",
    33: "aorta",
    41: "duodenum",
}
PORTAL_SOURCE_LABEL_IDS = frozenset({26, 27})
_POSSIBLE_ORGANS_SCHEMA_VERSION = "eus-possible-organs/v1"
_EUS_LABEL_NAMES = {
    1: "S1 第二肝门",
    2: "S2 肝脏",
    3: "S3 腹主动脉",
    4: "S4 胰体",
    5: "S5 胰尾",
    6: "S6 脾门",
    7: "S7 胰颈",
    8: "S8 第一肝门",
    9: "S9 胰头",
    10: "S10 壶腹部",
    11: "S11 钩突",
    14: "肝脏",
    18: "脾脏",
    19: "胰腺",
    20: "胰管",
    21: "壶腹部",
    22: "左侧肾上腺",
    23: "右侧肾上腺",
    24: "左侧肾脏",
    25: "右侧肾脏",
    26: "门静脉（包括分支",
    27: "门静脉汇合部",
    30: "下腔静脉",
    33: "腹主动脉",
    41: "十二指肠肠腔",
}
_ORGAN_VESSEL_ROLES = {
    "aorta": ("organ_and_vessel", "artery", None),
    "inferior_vena_cava": ("organ_and_vessel", "vein", None),
    "portal_vein": ("organ_and_vessel", "vein", 26),
}


@dataclass(frozen=True)
class _VesselSpec:
    label: str
    label_id: int
    source_label_ids: frozenset[int]


_VESSEL_SPECS = (
    _VesselSpec("vein", 26, PORTAL_SOURCE_LABEL_IDS),
    *(
        _VesselSpec("vein", label_id, frozenset({label_id}))
        for label_id in (28, 29, 30, 31, 32)
    ),
    *(
        _VesselSpec("artery", label_id, frozenset({label_id}))
        for label_id in (3, 33, 34, 35, 36, 37, 38, 39, 40)
    ),
)


@dataclass(frozen=True)
class CroppedFeatureResult:
    """Paths and gallery record produced for one cropped frame."""

    folder: Path
    feature_path: Path
    gallery_path: Path
    record: dict[str, Any]


def possible_organs_catalog() -> dict[str, Any]:
    """Return the fixed EUS organ vocabulary and its source-label mapping."""
    organs: list[dict[str, Any]] = []
    for organ_label in sorted(set(ORGAN_LABEL_BY_ID.values())):
        label_ids = sorted(
            label_id
            for label_id, mapped_label in ORGAN_LABEL_BY_ID.items()
            if mapped_label == organ_label
        )
        role, vessel_type, canonical_vessel_label_id = _ORGAN_VESSEL_ROLES.get(
            organ_label, ("organ", None, None)
        )
        organs.append(
            {
                "organ_label": organ_label,
                "eus_label_ids": label_ids,
                "eus_label_names": [
                    _EUS_LABEL_NAMES[label_id] for label_id in label_ids
                ],
                "role": role,
                "vessel_type": vessel_type,
                "canonical_vessel_label_id": canonical_vessel_label_id,
            }
        )
    return {"schema_version": _POSSIBLE_ORGANS_SCHEMA_VERSION, "organs": organs}


def write_possible_organs_catalog(directory: str | Path) -> Path:
    """Write the fixed possible-organ catalog into a batch output root."""
    path = Path(directory) / "eus_possible_organs.json"
    path.write_text(
        json.dumps(possible_organs_catalog(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _tar_members(tar_path: Path) -> tuple[dict[str, Any], bytes | None, str | None]:
    with tarfile.open(tar_path, "r:*") as archive:
        json_members = [
            member
            for member in archive.getmembers()
            if member.isfile() and member.name.lower().endswith(".json")
        ]
        nifti_members = [
            member
            for member in archive.getmembers()
            if member.isfile()
            and (
                member.name.lower().endswith(".nii.gz")
                or member.name.lower().endswith(".nii")
            )
        ]
        if len(json_members) != 1 or len(nifti_members) > 1:
            raise ValueError(
                "Label TAR must contain one JSON and at most one NIfTI: "
                f"{tar_path}"
            )

        json_stream = archive.extractfile(json_members[0])
        if json_stream is None:
            raise ValueError(f"Unable to read label JSON: {tar_path}")
        metadata = json.load(json_stream)

        if not nifti_members:
            return metadata, None, None

        nifti_member = nifti_members[0]
        nifti_stream = archive.extractfile(nifti_member)
        if nifti_stream is None:
            raise ValueError(f"Unable to read label NIfTI: {tar_path}")
        suffix = ".nii.gz" if nifti_member.name.lower().endswith(".nii.gz") else ".nii"
        return metadata, nifti_stream.read(), suffix


def _read_label_image(payload: bytes, suffix: str) -> np.ndarray:
    """Read NIfTI's [x, y] data as image-coordinate [y, x] labels."""
    with tempfile.TemporaryDirectory() as temporary_directory:
        path = Path(temporary_directory) / f"label{suffix}"
        path.write_bytes(payload)
        values_xy = np.asanyarray(nib.load(str(path)).dataobj).copy()

    if values_xy.ndim != 2:
        raise ValueError(
            "Cropped label NIfTI must be two-dimensional: " f"{values_xy.shape}"
        )
    return np.asarray(values_xy).T


def _empty_label_image(metadata: dict[str, Any]) -> np.ndarray:
    try:
        width = int(metadata["FileInfo"]["Width"])
        height = int(metadata["FileInfo"]["Height"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "JSON-only label TAR must provide FileInfo.Width and FileInfo.Height"
        ) from error
    if width < 2 or height < 2:
        raise ValueError(f"Invalid JSON-only label size: {width} x {height}")
    return np.zeros((height, width), dtype=np.uint16)


def _label_table(metadata: dict[str, Any]) -> dict[int, dict[str, Any]]:
    try:
        entries = metadata["Models"]["ColorLabelTableModel"]
    except (KeyError, TypeError) as error:
        raise ValueError("Label JSON is missing Models.ColorLabelTableModel") from error
    if not isinstance(entries, list):
        raise ValueError("ColorLabelTableModel must be a list")

    table: dict[int, dict[str, Any]] = {}
    for entry in entries:
        if isinstance(entry, dict) and isinstance(entry.get("ID"), int):
            label_id = entry["ID"]
            table[label_id] = {
                "description": str(entry.get("Desc", f"label_{label_id}"))
            }
    return table


def _organ_ids_from_frame_label(metadata: dict[str, Any]) -> list[int]:
    models = metadata.get("Models")
    frame_model = models.get("FrameLabelModel") if isinstance(models, dict) else None
    entries = frame_model.get("FrameLabel") if isinstance(frame_model, dict) else None
    if not isinstance(entries, list):
        return []

    return sorted(
        {
            entry["Label"]
            for entry in entries
            if isinstance(entry, dict)
            and isinstance(entry.get("Label"), int)
            and not isinstance(entry.get("Label"), bool)
            and entry["Label"] in ORGAN_LABEL_BY_ID
        }
    )


def _organ_ids_from_cropped_labels(labels: np.ndarray) -> list[int]:
    return sorted(
        {
            int(label_id)
            for label_id in np.unique(labels)
            if int(label_id) in ORGAN_LABEL_BY_ID
        }
    )


def _organ_metadata(metadata: dict[str, Any], labels: np.ndarray) -> dict[str, Any]:
    frame_ids = _organ_ids_from_frame_label(metadata)
    nifti_ids = _organ_ids_from_cropped_labels(labels)
    return {
        "organ_label_source": "frame_label_and_cropped_nifti",
        "frame_label_organ_ids": frame_ids,
        "cropped_nifti_organ_ids": nifti_ids,
        "organ_labels": sorted(
            {
                ORGAN_LABEL_BY_ID[label_id]
                for label_id in (*frame_ids, *nifti_ids)
            }
        ),
    }


def _features(
    labels: np.ndarray,
    table: dict[int, dict[str, Any]],
    pixel_spacing_mm: tuple[float, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return complete vessel components and edge-touching components separately."""
    height, width = labels.shape
    x_spacing, y_spacing = pixel_spacing_mm
    structure = np.ones((3, 3), dtype=np.uint8)
    features: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for spec in _VESSEL_SPECS:
        components, count = ndimage.label(
            np.isin(labels, tuple(spec.source_label_ids)), structure=structure
        )
        for component_index in range(1, count + 1):
            points_yx = np.argwhere(components == component_index)
            y_values = points_yx[:, 0]
            x_values = points_yx[:, 1]
            base = {
                "label": spec.label,
                "label_id": spec.label_id,
                "label_desc": table.get(spec.label_id, {}).get(
                    "description", f"label_{spec.label_id}"
                ),
                "component_index": component_index,
                "area_px": int(len(points_yx)),
                "centroid_px": [
                    float(np.mean(x_values)),
                    float(np.mean(y_values)),
                ],
            }
            if spec.source_label_ids == PORTAL_SOURCE_LABEL_IDS:
                base["source_label_ids"] = sorted(
                    {
                        int(label_id)
                        for label_id in np.unique(labels[y_values, x_values])
                        if int(label_id) in spec.source_label_ids
                    }
                )

            touches_edge = bool(
                np.any(x_values == 0)
                or np.any(x_values == width - 1)
                or np.any(y_values == 0)
                or np.any(y_values == height - 1)
            )
            if touches_edge:
                skipped.append({**base, "reason": "touches_image_edge"})
                continue

            features.append(
                {
                    **base,
                    "x_mm": float(np.mean(x_values) * x_spacing),
                    "y_mm": float(np.mean(y_values) * y_spacing),
                    "area_mm2": float(len(points_yx) * x_spacing * y_spacing),
                }
            )

    return features, skipped


def _gallery_record(
    stem: str,
    features: list[dict[str, Any]],
    width_mm: float,
    length_mm: float,
    pixel_spacing_mm: tuple[float, float],
    organ_metadata: dict[str, Any],
) -> dict[str, Any]:
    adapter_features = [
        {key: feature[key] for key in ("label", "x_mm", "y_mm", "area_mm2")}
        for feature in features
    ]
    center = [width_mm / 2.0, length_mm / 2.0, 0.0]
    return {
        "frame_id": stem,
        "slice_id": f"{stem}_cropped",
        "status": "gallery" if adapter_features else "unindexed",
        "organ": "unknown",
        "organ_label_source": organ_metadata["organ_label_source"],
        "organ_labels": organ_metadata["organ_labels"],
        "source": "cropped_label_tar",
        "probe_point_world": center,
        "input_normal_world": [0.0, 0.0, 1.0],
        "input_direction_world": [0.0, 1.0, 0.0],
        "square_vertices_world": [
            [0.0, 0.0, 0.0],
            [width_mm, 0.0, 0.0],
            [width_mm, length_mm, 0.0],
            [0.0, length_mm, 0.0],
        ],
        "origin_world": [0.0, 0.0, 0.0],
        "center_world": center,
        "u_axis_world": [1.0, 0.0, 0.0],
        "v_axis_world": [0.0, 1.0, 0.0],
        "normal_world": [0.0, 0.0, 1.0],
        "width_mm": width_mm,
        "length_mm": length_mm,
        "pixel_spacing_mm": list(pixel_spacing_mm),
        "ct_png": f"{stem}_cropped.jpg",
        "boundary_only_png": f"{stem}_cropped_vessel_label_white.png",
        "ct_overlay_png": f"{stem}_cropped_vessel_overlay.png",
        "features": adapter_features,
        "quality": {
            "accepted": True,
            "reason": None,
            "black_ratio": None,
            "line_length_px": None,
            "black_side_ratio": None,
            "valid_side_black_ratio": None,
        },
        "resampling_backend": "label_tar_2d",
        "pose_coordinate_system": "synthetic_2d_10cm_crop",
        "patient_world_pose": False,
    }


def _validate_visual_images(directory: Path, stem: str, image_size: tuple[int, int]) -> None:
    for suffix in ("_cropped_vessel_label_white.png", "_cropped_vessel_overlay.png"):
        path = directory / f"{stem}{suffix}"
        if not path.is_file():
            continue
        with Image.open(path) as image:
            if image.size != image_size:
                raise ValueError(
                    "Existing vessel visualization size does not match label image: "
                    f"{path}"
                )


def _write_artifacts(
    directory: Path,
    stem: str,
    labels: np.ndarray,
    label_source: str,
    features: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    width_mm: float,
    length_mm: float,
    pixel_spacing_mm: tuple[float, float],
    organ_metadata: dict[str, Any],
    record: dict[str, Any],
) -> CroppedFeatureResult:
    height, width = labels.shape
    feature_path = directory / f"{stem}_cropped_retrieval_features.json"
    gallery_path = directory / f"{stem}_cropped_gallery.jsonl"
    details = {
        "schema_version": _SCHEMA_VERSION,
        "frame_id": stem,
        "label_tar": f"{stem}_cropped_jpg_Label.tar",
        "label_source": label_source,
        "label_white_png": f"{stem}_cropped_vessel_label_white.png",
        "image_size_px": [width, height],
        "crop_size_mm": [width_mm, length_mm],
        "pixel_spacing_mm": list(pixel_spacing_mm),
        "feature_coordinate_system": "top_left_origin_x_right_y_down_mm",
        "organ_label_source": organ_metadata["organ_label_source"],
        "frame_label_organ_ids": organ_metadata["frame_label_organ_ids"],
        "cropped_nifti_organ_ids": organ_metadata["cropped_nifti_organ_ids"],
        "organ_labels": organ_metadata["organ_labels"],
        "features": features,
        "skipped_components": skipped,
        "adapter_record": record,
    }
    feature_path.write_text(
        json.dumps(details, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    gallery_path.write_text(
        json.dumps(record, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return CroppedFeatureResult(directory, feature_path, gallery_path, record)


def process_cropped_folder(
    folder: str | Path,
    width_mm: float = 100.0,
    length_mm: float = 100.0,
) -> CroppedFeatureResult:
    """Extract complete artery/vein sections and write per-frame retrieval files."""
    if width_mm <= 0.0 or length_mm <= 0.0:
        raise ValueError("Crop physical dimensions must be positive")

    directory = Path(folder)
    stem = directory.name
    tar_path = directory / f"{stem}_cropped_jpg_Label.tar"
    metadata, nifti_payload, suffix = _tar_members(tar_path)
    if nifti_payload is None:
        labels = _empty_label_image(metadata)
        label_source = "empty_label_json"
    else:
        if suffix is None:
            raise ValueError("NIfTI payload is missing its filename suffix")
        labels = _read_label_image(nifti_payload, suffix)
        label_source = "nifti"

    try:
        expected_width = int(metadata["FileInfo"]["Width"])
        expected_height = int(metadata["FileInfo"]["Height"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Label JSON must provide FileInfo.Width and FileInfo.Height") from error
    if labels.shape != (expected_height, expected_width):
        raise ValueError(
            "Label image size does not match label JSON: "
            f"{labels.shape[::-1]} != {(expected_width, expected_height)}"
        )
    if expected_width < 2 or expected_height < 2:
        raise ValueError(
            f"Invalid cropped label size: {expected_width} x {expected_height}"
        )

    _validate_visual_images(directory, stem, (expected_width, expected_height))
    pixel_spacing_mm = (
        width_mm / (expected_width - 1),
        length_mm / (expected_height - 1),
    )
    organ_metadata = _organ_metadata(metadata, labels)
    features, skipped = _features(labels, _label_table(metadata), pixel_spacing_mm)
    record = _gallery_record(
        stem, features, width_mm, length_mm, pixel_spacing_mm, organ_metadata
    )
    return _write_artifacts(
        directory,
        stem,
        labels,
        label_source,
        features,
        skipped,
        width_mm,
        length_mm,
        pixel_spacing_mm,
        organ_metadata,
        record,
    )
