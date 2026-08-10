"""End-to-end tests for the 10 cm picked-data batch script."""

import io
import json
import tarfile

import nibabel as nib
import numpy as np
from PIL import Image

from scripts.crop_picked_10cm import run_batch


def _add_tar_member(archive: tarfile.TarFile, name: str, content: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(content)
    archive.addfile(info, io.BytesIO(content))


def _label_data() -> dict:
    return {
        "FileInfo": {"Width": 1920, "Height": 1080, "Name": "frame_00000001.jpg"},
        "FileName": "frame_00000001_jpg",
        "Models": {
            "ColorLabelTableModel": [
                {"ID": 3, "Desc": "腹主动脉", "Color": [0, 170, 127, 255]},
                {"ID": 14, "Desc": "肝脏", "Color": [170, 0, 0, 255]},
                {"ID": 15, "Color": [0, 85, 0, 255]},
                {"ID": 18, "Desc": "脾脏", "Color": [233, 150, 122, 255]},
                {"ID": 26, "Color": [170, 85, 255, 255]},
                {"ID": 33, "Color": [255, 0, 0, 255]},
            ],
            "FrameLabelModel": {
                "FrameLabel": [
                    {"FrameCount": 0, "ItemType": 0, "Label": 14, "ViewType": 3},
                    {"FrameCount": 0, "ItemType": 0, "Label": 15, "ViewType": 3},
                ]
            },
            "PolygonModel2": None,
        },
        "Polys": [
            {
                "Shapes": [
                    {
                        "labelType": 26,
                        "Points": [
                            {"Pos": [603, 123, 0.0]},
                            {"Pos": [720, 123, 0.0]},
                            {"Pos": [720, 240, 0.0]},
                            {"Pos": [603, 240, 0.0]},
                        ],
                    },
                    {
                        "labelType": 15,
                        "Points": [
                            {"Pos": [900, 123, 0.0]},
                            {"Pos": [950, 123, 0.0]},
                            {"Pos": [950, 200, 0.0]},
                            {"Pos": [900, 200, 0.0]},
                        ],
                    },
                    {
                        "labelType": 33,
                        "Points": [
                            {"Pos": [750, 123, 0.0]},
                            {"Pos": [800, 123, 0.0]},
                            {"Pos": [800, 200, 0.0]},
                            {"Pos": [750, 200, 0.0]},
                        ],
                    }
                ]
            }
        ],
    }


def _source_nifti_bytes(tmp_path) -> bytes:
    labels_xy = np.zeros((1920, 1080), dtype=np.uint16)
    labels_xy[700:702, 200:202] = 26
    labels_xy[800:802, 300:302] = 3
    labels_xy[900:902, 250:252] = 18
    nifti_path = tmp_path / "source_labels.nii.gz"
    nib.Nifti1Image(labels_xy, np.eye(4)).to_filename(nifti_path)
    return nifti_path.read_bytes()


def test_run_batch_writes_retrieval_artifacts_with_vessel_only_visualizations(tmp_path):
    input_dir = tmp_path / "picked"
    output_dir = tmp_path / "picked_10cm_cropped"
    input_dir.mkdir()

    source_image = input_dir / "frame_00000001.jpg"
    Image.new("RGB", (1920, 1080), "navy").save(source_image, quality=100, subsampling=0)

    source_label = input_dir / "frame_00000001_jpg_Label.tar"
    with tarfile.open(source_label, "w") as archive:
        _add_tar_member(
            archive,
            "frame_00000001_jpg_Label.json",
            json.dumps(_label_data(), ensure_ascii=False).encode("utf-8"),
        )
        _add_tar_member(
            archive,
            "frame_00000001_jpg_Label.nii.gz",
            _source_nifti_bytes(tmp_path),
        )

    result = run_batch(input_dir, output_dir)

    assert result == {"total": 1, "processed": 1, "gallery_records": 1}
    frame_dir = output_dir / "frame_00000001"
    assert {path.name for path in frame_dir.iterdir()} == {
        "frame_00000001.jpg",
        "frame_00000001_jpg_Label.tar",
        "frame_00000001_cropped.jpg",
        "frame_00000001_cropped_jpg_Label.tar",
        "frame_00000001_cropped_label_white.png",
        "frame_00000001_original_overlay.png",
        "frame_00000001_cropped_overlay.png",
        "frame_00000001_cropped_vessel_overlay.png",
        "frame_00000001_cropped_vessel_label_white.png",
        "frame_00000001_cropped_retrieval_features.json",
        "frame_00000001_cropped_gallery.jsonl",
    }
    assert (frame_dir / source_image.name).read_bytes() == source_image.read_bytes()
    assert (frame_dir / source_label.name).read_bytes() == source_label.read_bytes()

    with Image.open(frame_dir / "frame_00000001_cropped.jpg") as cropped:
        assert cropped.size == (960, 960)
        assert np.all(np.asarray(cropped)[957:, :, :] >= 250)

    with Image.open(frame_dir / "frame_00000001_cropped_label_white.png") as labels:
        assert labels.size == (960, 960)
        assert np.any(np.asarray(labels) != 255)

    with Image.open(frame_dir / "frame_00000001_original_overlay.png") as original_overlay:
        assert original_overlay.size == (1920, 1080)
    with Image.open(frame_dir / "frame_00000001_cropped_overlay.png") as cropped_overlay:
        assert cropped_overlay.size == (960, 960)
    with Image.open(frame_dir / "frame_00000001_cropped_vessel_overlay.png") as vessel_overlay:
        assert vessel_overlay.size == (960, 960)
    with Image.open(frame_dir / "frame_00000001_cropped_vessel_label_white.png") as vessel_labels:
        assert vessel_labels.size == (960, 960)
        assert vessel_labels.getpixel((0, 50)) == (0, 188, 212)
        assert vessel_labels.getpixel((147, 50)) == (255, 82, 0)
        assert vessel_labels.getpixel((297, 50)) == (255, 255, 255)
        colors = {
            tuple(color)
            for color in np.unique(np.asarray(vessel_labels).reshape(-1, 3), axis=0)
        }
        assert colors <= {(255, 255, 255), (255, 82, 0), (0, 188, 212)}

    details = json.loads(
        (frame_dir / "frame_00000001_cropped_retrieval_features.json").read_text(
            encoding="utf-8"
        )
    )
    assert [feature["label_id"] for feature in details["features"]] == [26, 3]
    assert details["frame_label_organ_ids"] == [14]
    assert details["cropped_nifti_organ_ids"] == [3, 18, 26]
    assert details["organ_labels"] == [
        "aorta",
        "liver",
        "portal_vein",
        "spleen",
    ]

    gallery_record = json.loads(
        (frame_dir / "frame_00000001_cropped_gallery.jsonl").read_text(
            encoding="utf-8"
        )
    )
    assert gallery_record["status"] == "gallery"
    assert gallery_record["boundary_only_png"] == (
        "frame_00000001_cropped_vessel_label_white.png"
    )
    assert gallery_record["ct_overlay_png"] == "frame_00000001_cropped_vessel_overlay.png"
    assert gallery_record["organ_label_source"] == "frame_label_and_cropped_nifti"
    assert gallery_record["organ_labels"] == details["organ_labels"]
    assert not (output_dir / "retrieval_gallery.jsonl").exists()
    assert not (output_dir / "retrieval_feature_summary.json").exists()

    catalog = json.loads(
        (output_dir / "eus_possible_organs.json").read_text(encoding="utf-8")
    )
    assert catalog["schema_version"] == "eus-possible-organs/v1"
    assert len(catalog["organs"]) == 11
    by_name = {item["organ_label"]: item for item in catalog["organs"]}
    assert by_name["aorta"]["eus_label_ids"] == [3, 33]
    assert by_name["aorta"]["role"] == "organ_and_vessel"
    assert by_name["aorta"]["vessel_type"] == "artery"
    assert by_name["portal_vein"]["eus_label_ids"] == [26, 27]
    assert by_name["portal_vein"]["canonical_vessel_label_id"] == 26
    assert by_name["liver"]["role"] == "organ"

    with tarfile.open(frame_dir / "frame_00000001_cropped_jpg_Label.tar", "r") as archive:
        assert archive.getnames() == [
            "frame_00000001_cropped_jpg_Label.json",
            "frame_00000001_cropped_jpg_Label.nii.gz",
        ]
