"""End-to-end tests for the 10 cm picked-data batch script."""

import io
import json
import tarfile

import nibabel as nib
import numpy as np
from PIL import Image

import scripts.crop_picked_10cm as crop_picked_10cm
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
                {"ID": 27, "Color": [170, 85, 255, 255]},
                {"ID": 28, "Color": [170, 85, 255, 255]},
                {"ID": 29, "Color": [170, 85, 255, 255]},
                {"ID": 30, "Color": [0, 0, 255, 255]},
                {"ID": 31, "Color": [0, 188, 212, 255]},
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
                    },
                    {
                        "labelType": 3,
                        "Points": [
                            {"Pos": [650, 180, 0.0]},
                            {"Pos": [670, 180, 0.0]},
                            {"Pos": [670, 200, 0.0]},
                            {"Pos": [650, 200, 0.0]},
                        ],
                    },
                    {
                        "labelType": 30,
                        "Points": [
                            {"Pos": [730, 180, 0.0]},
                            {"Pos": [750, 180, 0.0]},
                            {"Pos": [750, 200, 0.0]},
                            {"Pos": [730, 200, 0.0]},
                        ],
                    },
                    {
                        "labelType": 27,
                        "Points": [
                            {"Pos": [790, 180, 0.0]},
                            {"Pos": [810, 180, 0.0]},
                            {"Pos": [810, 200, 0.0]},
                            {"Pos": [790, 200, 0.0]},
                        ],
                    },
                    {
                        "labelType": 28,
                        "Points": [
                            {"Pos": [830, 180, 0.0]},
                            {"Pos": [850, 180, 0.0]},
                            {"Pos": [850, 200, 0.0]},
                            {"Pos": [830, 200, 0.0]},
                        ],
                    },
                    {
                        "labelType": 29,
                        "Points": [
                            {"Pos": [870, 180, 0.0]},
                            {"Pos": [890, 180, 0.0]},
                            {"Pos": [890, 200, 0.0]},
                            {"Pos": [870, 200, 0.0]},
                        ],
                    },
                    {
                        "labelType": 31,
                        "Points": [
                            {"Pos": [910, 180, 0.0]},
                            {"Pos": [930, 180, 0.0]},
                            {"Pos": [930, 200, 0.0]},
                            {"Pos": [910, 200, 0.0]},
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
    labels_xy[730:732, 180:182] = 30
    labels_xy[790:792, 180:182] = 27
    labels_xy[830:832, 180:182] = 28
    labels_xy[870:872, 180:182] = 29
    labels_xy[910:912, 180:182] = 31
    nifti_path = tmp_path / "source_labels.nii.gz"
    nib.Nifti1Image(labels_xy, np.eye(4)).to_filename(nifti_path)
    return nifti_path.read_bytes()


def test_draw_anatomical_vessel_outlines_uses_fixed_three_class_palette():
    def shape(label_id: int, x: int, y: int) -> dict:
        return {
            "labelType": label_id,
            "Points": [
                {"Pos": [x, y, 0.0]},
                {"Pos": [x + 2, y, 0.0]},
                {"Pos": [x + 2, y + 2, 0.0]},
                {"Pos": [x, y + 2, 0.0]},
            ],
        }

    label_data = {
        "Models": {"PolygonModel2": None},
        "Polys": [
            {
                "Shapes": [
                    shape(3, 2, 3),
                    shape(30, 7, 3),
                    shape(26, 12, 2),
                    shape(27, 12, 6),
                    shape(28, 12, 10),
                    shape(29, 12, 14),
                    shape(31, 17, 3),
                ]
            }
        ],
    }

    result = crop_picked_10cm.draw_anatomical_vessel_outlines(
        Image.new("RGB", (20, 20), "white"), label_data
    )

    assert result.getpixel((2, 3)) == (255, 0, 0)
    assert result.getpixel((7, 3)) == (0, 0, 255)
    assert {result.getpixel((12, y)) for y in (2, 6, 10, 14)} == {
        (170, 85, 255)
    }
    assert result.getpixel((17, 3)) == (255, 255, 255)


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
        "frame_00000001_original_ivc_ao_pv_overlay.png",
        "frame_00000001_cropped_ivc_ao_pv_overlay.png",
        "frame_00000001_cropped_ivc_ao_pv_label_white.png",
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
    with Image.open(frame_dir / "frame_00000001_original_ivc_ao_pv_overlay.png") as image:
        assert image.size == (1920, 1080)
    with Image.open(frame_dir / "frame_00000001_cropped_ivc_ao_pv_overlay.png") as image:
        assert image.size == (960, 960)
    with Image.open(frame_dir / "frame_00000001_cropped_ivc_ao_pv_label_white.png") as image:
        assert image.size == (960, 960)
        assert image.getpixel((0, 50)) == (170, 85, 255)
        assert image.getpixel((47, 57)) == (255, 0, 0)
        assert image.getpixel((127, 57)) == (0, 0, 255)
        assert {
            image.getpixel((x, 57)) for x in (187, 227, 267)
        } == {(170, 85, 255)}
        colors = {
            tuple(color)
            for color in np.unique(np.asarray(image).reshape(-1, 3), axis=0)
        }
        assert colors <= {
            (255, 255, 255),
            (255, 0, 0),
            (0, 0, 255),
            (170, 85, 255),
        }
        assert {(255, 0, 0), (0, 0, 255), (170, 85, 255)} <= colors

    details = json.loads(
        (frame_dir / "frame_00000001_cropped_retrieval_features.json").read_text(
            encoding="utf-8"
        )
    )
    assert details["schema_version"] == "cropped-retrieval-features/v2"
    assert [feature["label_id"] for feature in details["features"]] == [
        26,
        26,
        28,
        29,
        30,
        31,
        3,
    ]
    assert details["anatomical_vessel_visualizations"] == {
        "original_overlay_png": "frame_00000001_original_ivc_ao_pv_overlay.png",
        "cropped_overlay_png": "frame_00000001_cropped_ivc_ao_pv_overlay.png",
        "boundary_only_png": "frame_00000001_cropped_ivc_ao_pv_label_white.png",
    }
    assert {
        (feature["label"], tuple(feature["source_label_ids"]))
        for feature in details["anatomical_vessel_features"]
    } == {
        ("aorta", (3,)),
        ("inferior_vena_cava", (30,)),
        ("portal_venous_system", (26,)),
        ("portal_venous_system", (27,)),
        ("portal_venous_system", (28,)),
        ("portal_venous_system", (29,)),
    }
    assert details["anatomical_vessel_skipped_components"] == []
    assert details["frame_label_organ_ids"] == [14]
    assert details["cropped_nifti_organ_ids"] == [3, 18, 26, 27, 30]
    assert details["organ_labels"] == [
        "aorta",
        "inferior_vena_cava",
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
