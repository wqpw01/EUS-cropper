"""Behavioral tests for cropped vessel retrieval feature extraction."""

import io
import json
import tarfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from src.cropped_retrieval import (
    ARTERY_LABEL_IDS,
    VEIN_LABEL_IDS,
    process_cropped_folder,
)


FRAME_ID = "frame_00000001"
LABEL_TAR_NAME = f"{FRAME_ID}_cropped_jpg_Label.tar"
DETAILS_NAME = f"{FRAME_ID}_cropped_retrieval_features.json"
GALLERY_NAME = f"{FRAME_ID}_cropped_gallery.jsonl"


def _add_tar_member(archive: tarfile.TarFile, name: str, content: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(content)
    archive.addfile(member, io.BytesIO(content))


def _label_data(frame_label_ids: list[int] | None = None) -> dict:
    return {
        "FileInfo": {"Width": 10, "Height": 10},
        "Models": {
            "ColorLabelTableModel": [
                {"ID": 2, "Desc": "S2 liver"},
                {"ID": 3, "Desc": "artery"},
                {"ID": 12, "Desc": "gallbladder"},
                {"ID": 26, "Desc": "vein"},
                {"ID": 27, "Desc": "portal confluence"},
                {"ID": 28, "Desc": "mesenteric vein"},
                {"ID": 29, "Desc": "splenic vein"},
                {"ID": 30, "Desc": "inferior vena cava"},
                {"ID": 32, "Desc": "vein"},
                {"ID": 33, "Desc": "artery"},
                {"ID": 40, "Desc": "artery"},
                {"ID": 16, "Desc": "bile duct"},
                {"ID": 18, "Desc": "spleen"},
                {"ID": 19, "Desc": "pancreas"},
                {"ID": 15, "Desc": "nonvessel"},
            ],
            "FrameLabelModel": {
                "FrameLabel": [
                    {
                        "FrameCount": 0,
                        "ItemType": 0,
                        "Label": label_id,
                        "ViewType": 3,
                    }
                    for label_id in frame_label_ids or []
                ]
            },
        },
    }


@pytest.fixture
def cropped_label_tar(tmp_path: Path):
    def write(
        labels_xy: np.ndarray | None = None,
        frame_label_ids: list[int] | None = None,
    ) -> Path:
        frame_dir = tmp_path / FRAME_ID
        frame_dir.mkdir()
        tar_path = frame_dir / LABEL_TAR_NAME
        with tarfile.open(tar_path, "w") as archive:
            _add_tar_member(
                archive,
                f"{FRAME_ID}_cropped_jpg_Label.json",
                json.dumps(_label_data(frame_label_ids)).encode("utf-8"),
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


def test_vessel_label_groups_match_the_fixed_crop_definition():
    assert VEIN_LABEL_IDS == frozenset({26, 27, 28, 29, 30, 31, 32})
    assert ARTERY_LABEL_IDS == frozenset({3, 33, 34, 35, 36, 37, 38, 39, 40})


def test_process_cropped_folder_extracts_complete_vessel_components(cropped_label_tar):
    labels_xy = np.zeros((10, 10), dtype=np.uint16)
    # The NIfTI fixture uses [x, y] axis order.
    labels_xy[2:4, 5:7] = 26
    labels_xy[6:8, 2:4] = 3
    labels_xy[0:2, 4:6] = 33
    labels_xy[4:6, 7:9] = 15
    frame_dir = cropped_label_tar(labels_xy)

    process_cropped_folder(frame_dir)

    details_path = frame_dir / DETAILS_NAME
    gallery_path = frame_dir / GALLERY_NAME
    assert details_path.is_file()
    assert gallery_path.is_file()

    details = json.loads(details_path.read_text(encoding="utf-8"))
    assert {
        "schema_version",
        "frame_id",
        "label_tar",
        "label_source",
        "label_white_png",
        "image_size_px",
        "crop_size_mm",
        "pixel_spacing_mm",
        "feature_coordinate_system",
        "organ_label_source",
        "frame_label_organ_ids",
        "cropped_nifti_organ_ids",
        "organ_labels",
        "features",
        "skipped_components",
        "adapter_record",
    } <= set(details)
    assert details["frame_id"] == FRAME_ID
    assert details["label_tar"] == LABEL_TAR_NAME
    assert details["label_source"] == "nifti"
    assert details["feature_coordinate_system"] == "top_left_origin_x_right_y_down_mm"

    features = details["features"]
    assert [(feature["label"], feature["label_id"]) for feature in features] == [
        ("vein", 26),
        ("artery", 3),
    ]
    assert set(features[0]) == {
        "label",
        "label_id",
        "label_desc",
        "component_index",
        "area_px",
        "centroid_px",
        "x_mm",
        "y_mm",
        "area_mm2",
        "source_label_ids",
    }
    assert features[0]["source_label_ids"] == [26]
    assert set(features[1]) == {
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


def test_process_cropped_folder_uses_eight_connectivity_and_skips_every_edge(
    cropped_label_tar,
):
    labels_xy = np.zeros((10, 10), dtype=np.uint16)
    labels_xy[2, 2] = 32
    labels_xy[3, 3] = 32
    labels_xy[0, 5] = 40
    labels_xy[9, 5] = 40
    labels_xy[5, 0] = 40
    labels_xy[5, 9] = 40
    frame_dir = cropped_label_tar(labels_xy)

    process_cropped_folder(frame_dir)

    details = json.loads((frame_dir / DETAILS_NAME).read_text(encoding="utf-8"))
    assert [(feature["label"], feature["label_id"], feature["area_px"]) for feature in details["features"]] == [
        ("vein", 32, 2),
    ]
    assert details["features"][0]["centroid_px"] == [2.5, 2.5]
    assert len(details["skipped_components"]) == 4
    assert {item["label_id"] for item in details["skipped_components"]} == {40}
    assert {item["reason"] for item in details["skipped_components"]} == {
        "touches_image_edge"
    }


def test_process_cropped_folder_unions_frame_labels_and_cropped_nifti_organs(
    cropped_label_tar,
):
    labels_xy = np.zeros((10, 10), dtype=np.uint16)
    labels_xy[1:3, 1:3] = 18
    labels_xy[4:6, 1:3] = 30
    labels_xy[7:9, 1:3] = 15
    labels_xy[1:3, 6:8] = 28
    frame_dir = cropped_label_tar(
        labels_xy,
        frame_label_ids=[2, 3, 12, 16, 26, 27, 28],
    )

    process_cropped_folder(frame_dir)

    details = json.loads((frame_dir / DETAILS_NAME).read_text(encoding="utf-8"))
    assert details["organ_label_source"] == "frame_label_and_cropped_nifti"
    assert details["frame_label_organ_ids"] == [2, 3, 26, 27]
    assert details["cropped_nifti_organ_ids"] == [18, 30]
    assert details["organ_labels"] == [
        "aorta",
        "inferior_vena_cava",
        "liver",
        "portal_vein",
        "spleen",
    ]

    record = _read_gallery_record(frame_dir / GALLERY_NAME)
    assert record["organ"] == "unknown"
    assert record["organ_label_source"] == "frame_label_and_cropped_nifti"
    assert record["organ_labels"] == details["organ_labels"]


def test_process_cropped_folder_merges_portal_vein_and_confluence_components(
    cropped_label_tar,
):
    labels_xy = np.zeros((10, 10), dtype=np.uint16)
    labels_xy[2, 2] = 26
    labels_xy[3, 3] = 27
    labels_xy[0, 5] = 27
    frame_dir = cropped_label_tar(labels_xy, frame_label_ids=[27])

    process_cropped_folder(frame_dir)

    details = json.loads((frame_dir / DETAILS_NAME).read_text(encoding="utf-8"))
    portal = [item for item in details["features"] if item["label_id"] == 26]
    assert len(portal) == 1
    assert portal[0]["label"] == "vein"
    assert portal[0]["area_px"] == 2
    assert portal[0]["source_label_ids"] == [26, 27]
    skipped = [item for item in details["skipped_components"] if item["label_id"] == 26]
    assert len(skipped) == 1
    assert skipped[0]["source_label_ids"] == [27]
    assert details["organ_labels"] == ["portal_vein"]


def test_process_cropped_folder_writes_anatomical_vessel_features(cropped_label_tar):
    labels_xy = np.zeros((10, 10), dtype=np.uint16)
    labels_xy[2, 2] = 3
    labels_xy[3, 3] = 33
    labels_xy[5:7, 2:4] = 30
    labels_xy[2, 6] = 26
    labels_xy[3, 6] = 27
    labels_xy[3, 7] = 28
    labels_xy[4, 7] = 29
    labels_xy[0, 8] = 29
    frame_dir = cropped_label_tar(labels_xy)

    process_cropped_folder(frame_dir)

    details = json.loads((frame_dir / DETAILS_NAME).read_text(encoding="utf-8"))
    assert details["schema_version"] == "cropped-retrieval-features/v2"
    assert details["anatomical_vessel_visualizations"] == {
        "original_overlay_png": f"{FRAME_ID}_original_ivc_ao_pv_overlay.png",
        "cropped_overlay_png": f"{FRAME_ID}_cropped_ivc_ao_pv_overlay.png",
        "boundary_only_png": f"{FRAME_ID}_cropped_ivc_ao_pv_label_white.png",
    }
    assert [
        (item["label"], item["label_id"], item["source_label_ids"], item["area_px"])
        for item in details["anatomical_vessel_features"]
    ] == [
        ("aorta", 3, [3, 33], 2),
        ("inferior_vena_cava", 30, [30], 4),
        ("portal_venous_system", 26, [26, 27, 28, 29], 4),
    ]
    assert details["anatomical_vessel_skipped_components"] == [
        {
            "label": "portal_venous_system",
            "label_id": 26,
            "label_desc": "门静脉系",
            "component_index": 2,
            "area_px": 1,
            "centroid_px": [0.0, 8.0],
            "source_label_ids": [29],
            "reason": "touches_image_edge",
        }
    ]
    assert {item["label"] for item in details["features"]} <= {"vein", "artery"}
    assert {item["label"] for item in details["adapter_record"]["features"]} <= {
        "vein",
        "artery",
    }


def test_process_cropped_folder_writes_unindexed_record_without_nifti(cropped_label_tar):
    frame_dir = cropped_label_tar(frame_label_ids=[19, 15, 16, 28])

    process_cropped_folder(frame_dir)

    details_path = frame_dir / DETAILS_NAME
    gallery_path = frame_dir / GALLERY_NAME
    assert details_path.is_file()
    assert gallery_path.is_file()

    details = json.loads(details_path.read_text(encoding="utf-8"))
    assert details["label_source"] == "empty_label_json"
    assert details["features"] == []
    assert details["skipped_components"] == []
    assert details["frame_label_organ_ids"] == [19]
    assert details["cropped_nifti_organ_ids"] == []
    assert details["organ_labels"] == ["pancreas"]

    gallery_record = _read_gallery_record(gallery_path)
    assert gallery_record == details["adapter_record"]
    assert gallery_record["status"] == "unindexed"
    assert gallery_record["features"] == []
    assert gallery_record["organ_labels"] == ["pancreas"]
