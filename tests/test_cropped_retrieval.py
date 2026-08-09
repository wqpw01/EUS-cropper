"""Behavioral tests for cropped vessel retrieval feature extraction."""

import io
import json
import tarfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from src.cropped_retrieval import process_cropped_folder


FRAME_ID = "frame_00000001"
LABEL_TAR_NAME = f"{FRAME_ID}_cropped_jpg_Label.tar"
DETAILS_NAME = f"{FRAME_ID}_cropped_retrieval_features.json"
GALLERY_NAME = f"{FRAME_ID}_cropped_gallery.jsonl"


def _add_tar_member(archive: tarfile.TarFile, name: str, content: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(content)
    archive.addfile(member, io.BytesIO(content))


def _label_data() -> dict:
    return {
        "FileInfo": {"Width": 10, "Height": 10},
        "Models": {
            "ColorLabelTableModel": [
                {"ID": 3, "Desc": "artery"},
                {"ID": 26, "Desc": "vein"},
                {"ID": 33, "Desc": "artery"},
                {"ID": 15, "Desc": "nonvessel"},
            ]
        },
    }


@pytest.fixture
def cropped_label_tar(tmp_path: Path):
    def write(labels_xy: np.ndarray | None = None) -> Path:
        frame_dir = tmp_path / FRAME_ID
        frame_dir.mkdir()
        tar_path = frame_dir / LABEL_TAR_NAME
        with tarfile.open(tar_path, "w") as archive:
            _add_tar_member(
                archive,
                f"{FRAME_ID}_cropped_jpg_Label.json",
                json.dumps(_label_data()).encode("utf-8"),
            )
            if labels_xy is not None:
                nifti_path = tmp_path / "fixture_labels.nii.gz"
                nib.save(nib.Nifti1Image(labels_xy, np.eye(4)), str(nifti_path))
                _add_tar_member(
                    archive,
                    f"{FRAME_ID}_cropped_jpg_Label.nii.gz",
                    nifti_path.read_bytes(),
                )
        return frame_dir

    return write


def _read_gallery_record(path: Path) -> dict:
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    return json.loads(lines[0])


def test_process_cropped_folder_extracts_complete_vessel_components(cropped_label_tar):
    labels_xy = np.zeros((10, 10), dtype=np.uint16)
    # The NIfTI fixture uses [x, y] axis order.
    labels_xy[2:4, 5:7] = 26
    labels_xy[6:8, 2:4] = 3
    labels_xy[0:2, 4:6] = 33
    labels_xy[4:6, 7:9] = 15
    frame_dir = cropped_label_tar(labels_xy)

    process_cropped_folder(frame_dir, width_mm=100.0, length_mm=100.0)

    details_path = frame_dir / DETAILS_NAME
    gallery_path = frame_dir / GALLERY_NAME
    assert details_path.is_file()
    assert gallery_path.is_file()

    details = json.loads(details_path.read_text(encoding="utf-8"))
    assert set(details) == {
        "schema_version",
        "frame_id",
        "label_tar",
        "label_source",
        "label_white_png",
        "image_size_px",
        "crop_size_mm",
        "pixel_spacing_mm",
        "feature_coordinate_system",
        "features",
        "skipped_components",
        "adapter_record",
    }
    assert details["frame_id"] == FRAME_ID
    assert details["label_tar"] == LABEL_TAR_NAME
    assert details["label_source"] == "nifti"
    assert details["feature_coordinate_system"] == "top_left_origin_x_right_y_down_mm"

    features = details["features"]
    assert [(feature["label"], feature["label_id"]) for feature in features] == [
        ("vein", 26),
        ("artery", 3),
    ]
    assert all(
        set(feature)
        == {
            "label",
            "label_id",
            "label_desc",
            "component_index",
            "area_px",
            "centroid_px",
            "x_mm",
            "y_mm",
            "area_mm2",
        }
        for feature in features
    )

    spacing_mm = 100.0 / 9
    vein = features[0]
    assert vein["label_desc"] == "vein"
    assert vein["area_px"] == 4
    assert vein["centroid_px"] == [2.5, 5.5]
    assert vein["x_mm"] == pytest.approx(2.5 * spacing_mm)
    assert vein["y_mm"] == pytest.approx(5.5 * spacing_mm)
    assert vein["area_mm2"] == pytest.approx(4 * spacing_mm * spacing_mm)

    skipped = details["skipped_components"]
    assert len(skipped) == 1
    assert set(skipped[0]) == {
        "label",
        "label_id",
        "label_desc",
        "component_index",
        "area_px",
        "centroid_px",
        "x_mm",
        "y_mm",
        "area_mm2",
        "reason",
    }
    assert skipped[0]["label"] == "artery"
    assert skipped[0]["label_id"] == 33
    assert skipped[0]["label_desc"] == "artery"
    assert skipped[0]["area_px"] == 4
    assert skipped[0]["centroid_px"] == [0.5, 4.5]
    assert skipped[0]["reason"] == "touches_image_edge"

    gallery_record = _read_gallery_record(gallery_path)
    assert gallery_record == details["adapter_record"]
    assert gallery_record["status"] == "gallery"
    assert gallery_record["boundary_only_png"] == (
        f"{FRAME_ID}_cropped_vessel_label_white.png"
    )
    assert gallery_record["ct_overlay_png"] == f"{FRAME_ID}_cropped_vessel_overlay.png"
    assert [feature["label"] for feature in gallery_record["features"]] == [
        "vein",
        "artery",
    ]
    assert all(
        set(feature) == {"label", "x_mm", "y_mm", "area_mm2"}
        for feature in gallery_record["features"]
    )
    assert [feature["x_mm"] for feature in gallery_record["features"]] == pytest.approx(
        [2.5 * spacing_mm, 6.5 * spacing_mm]
    )
    assert [feature["y_mm"] for feature in gallery_record["features"]] == pytest.approx(
        [5.5 * spacing_mm, 2.5 * spacing_mm]
    )
    assert [feature["area_mm2"] for feature in gallery_record["features"]] == pytest.approx(
        [4 * spacing_mm * spacing_mm, 4 * spacing_mm * spacing_mm]
    )


def test_process_cropped_folder_writes_unindexed_record_without_nifti(cropped_label_tar):
    frame_dir = cropped_label_tar()

    process_cropped_folder(frame_dir)

    details_path = frame_dir / DETAILS_NAME
    gallery_path = frame_dir / GALLERY_NAME
    assert details_path.is_file()
    assert gallery_path.is_file()

    details = json.loads(details_path.read_text(encoding="utf-8"))
    assert details["label_source"] == "empty_label_json"
    assert details["features"] == []
    assert details["skipped_components"] == []

    gallery_record = _read_gallery_record(gallery_path)
    assert gallery_record == details["adapter_record"]
    assert gallery_record["status"] == "unindexed"
    assert gallery_record["features"] == []
