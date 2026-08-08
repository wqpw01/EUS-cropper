"""Tests for the fixed 10 cm crop core."""

import io
import json
import tarfile

import nibabel as nib
import numpy as np
from PIL import Image

from src.picked_10cm import (
    CROP_BOUNDS,
    CROP_SIZE,
    crop_image_to_canvas,
    crop_label_tar,
)


def _add_tar_member(archive: tarfile.TarFile, name: str, content: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(content)
    archive.addfile(info, io.BytesIO(content))


def _create_label_data() -> dict:
    return {
        "FileInfo": {
            "Width": 1920,
            "Height": 1080,
            "Name": "frame_00000001.jpg",
        },
        "FileName": "frame_00000001_jpg",
        "Description": "胰腺标注",
        "Models": {
            "PolygonModel2": [
                {
                    "Label": 7,
                    "Points": [[603, 123, 1.0], [1562, 1079, 1.0]],
                }
            ]
        },
    }


def _create_nifti_fixture(path):
    x_indices, y_indices = np.indices((1920, 1080))
    data = (x_indices * 20 + y_indices).astype(np.uint16)

    qform = np.array(
        [[0.5, 0.0, 0.0, 11.0], [0.0, 1.5, 0.0, 22.0], [0.0, 0.0, 2.5, 33.0], [0.0, 0.0, 0.0, 1.0]]
    )
    sform = np.array(
        [[0.75, 0.0, 0.0, 101.0], [0.0, 1.25, 0.0, 202.0], [0.0, 0.0, 2.0, 303.0], [0.0, 0.0, 0.0, 1.0]]
    )
    image = nib.Nifti1Image(data, sform)
    image.set_qform(qform, code=2)
    image.set_sform(sform, code=3)
    image.to_filename(path)
    return data, qform, sform


def _crop_translation() -> np.ndarray:
    translation = np.eye(4)
    translation[0, 3] = CROP_BOUNDS[0]
    translation[1, 3] = CROP_BOUNDS[1]
    return translation


def test_crop_image_to_canvas_copies_source_region_and_keeps_white_padding(tmp_path):
    source_path = tmp_path / "source.jpg"
    source = Image.new("RGB", (1920, 1080), "black")
    source.putpixel((603, 123), (210, 30, 40))
    source.putpixel((1562, 1079), (50, 170, 90))
    source.save(source_path, quality=100, subsampling=0)

    with Image.open(source_path) as reloaded_source:
        cropped = crop_image_to_canvas(reloaded_source)
        source_start = reloaded_source.getpixel((603, 123))
        source_end = reloaded_source.getpixel((1562, 1079))

    assert CROP_BOUNDS == (603, 123, 1563, 1083)
    assert cropped.mode == "RGB"
    assert cropped.size == CROP_SIZE == (960, 960)
    assert cropped.getpixel((0, 0)) == source_start
    assert cropped.getpixel((959, 956)) == source_end
    assert np.all(np.asarray(cropped)[957:, :, :] == 255)


def test_crop_label_tar_repackages_transformed_json_and_nifti(tmp_path):
    source_nifti_path = tmp_path / "source.nii.gz"
    source_data, source_qform, source_sform = _create_nifti_fixture(source_nifti_path)
    input_tar = tmp_path / "frame_00000001_jpg_Label.tar"
    output_tar = tmp_path / "frame_00000001_cropped_jpg_Label.tar"

    with tarfile.open(input_tar, "w") as archive:
        _add_tar_member(
            archive,
            "incoming/label.json",
            json.dumps(_create_label_data(), ensure_ascii=False).encode("utf-8"),
        )
        _add_tar_member(archive, "incoming/source-mask.nii.gz", source_nifti_path.read_bytes())

    crop_label_tar(
        input_tar,
        output_tar,
        cropped_jpg_filename="frame_00000001_cropped.jpg",
        json_member_basename="frame_00000001_cropped_jpg_Label.json",
        nifti_member_basename="frame_00000001_cropped_jpg_Label.nii.gz",
    )

    with tarfile.open(output_tar, "r") as archive:
        assert archive.getnames() == [
            "frame_00000001_cropped_jpg_Label.json",
            "frame_00000001_cropped_jpg_Label.nii.gz",
        ]
        json_content = archive.extractfile(archive.getmember(archive.getnames()[0])).read()
        output_label = json.loads(json_content.decode("utf-8"))
        nifti_content = archive.extractfile(archive.getmember(archive.getnames()[1])).read()

    assert "胰腺标注" in json_content.decode("utf-8")
    assert output_label["FileInfo"] == {
        "Width": 960,
        "Height": 960,
        "Name": "frame_00000001_cropped.jpg",
    }
    assert output_label["FileName"] == "frame_00000001_cropped_jpg"
    assert output_label["Models"]["PolygonModel2"][0]["Points"] == [
        [0, 0, 1.0],
        [959, 956, 1.0],
    ]

    output_nifti_path = tmp_path / "cropped.nii.gz"
    output_nifti_path.write_bytes(nifti_content)
    output_nifti = nib.load(output_nifti_path)
    output_data = np.asanyarray(output_nifti.dataobj)

    assert output_nifti.shape == (960, 960)
    assert output_nifti.get_data_dtype() == np.dtype(np.uint16)
    np.testing.assert_array_equal(output_data[:, :957], source_data[603:1563, 123:1080])
    assert np.all(output_data[:, 957:] == 0)
    np.testing.assert_allclose(output_nifti.get_qform(), source_qform @ _crop_translation())
    np.testing.assert_allclose(output_nifti.get_sform(), source_sform @ _crop_translation())
    np.testing.assert_allclose(output_nifti.affine, source_sform @ _crop_translation())
    assert output_nifti.get_qform(coded=True)[1] == 2
    assert output_nifti.get_sform(coded=True)[1] == 3


def test_crop_label_tar_repackages_uncompressed_nifti(tmp_path):
    source_nifti_path = tmp_path / "source.nii"
    source_data, _, _ = _create_nifti_fixture(source_nifti_path)
    input_tar = tmp_path / "frame_00000003_jpg_Label.tar"
    output_tar = tmp_path / "frame_00000003_cropped_jpg_Label.tar"

    with tarfile.open(input_tar, "w") as archive:
        _add_tar_member(
            archive,
            "source-label.json",
            json.dumps(_create_label_data(), ensure_ascii=False).encode("utf-8"),
        )
        _add_tar_member(archive, "source-mask.nii", source_nifti_path.read_bytes())

    crop_label_tar(
        input_tar,
        output_tar,
        cropped_jpg_filename="frame_00000003_cropped.jpg",
        json_member_basename="frame_00000003_cropped_jpg_Label.json",
        nifti_member_basename="frame_00000003_cropped_jpg_Label.nii",
    )

    with tarfile.open(output_tar, "r") as archive:
        assert archive.getnames() == [
            "frame_00000003_cropped_jpg_Label.json",
            "frame_00000003_cropped_jpg_Label.nii",
        ]
        nifti_content = archive.extractfile(archive.getnames()[1]).read()

    output_nifti_path = tmp_path / "cropped.nii"
    output_nifti_path.write_bytes(nifti_content)
    output_data = np.asanyarray(nib.load(output_nifti_path).dataobj)
    np.testing.assert_array_equal(output_data[:, :957], source_data[603:1563, 123:1080])
    assert np.all(output_data[:, 957:] == 0)


def test_crop_label_tar_without_nifti_writes_only_the_supplied_json_member(tmp_path):
    input_tar = tmp_path / "frame_00000002_jpg_Label.tar"
    output_tar = tmp_path / "frame_00000002_cropped_jpg_Label.tar"
    with tarfile.open(input_tar, "w") as archive:
        _add_tar_member(
            archive,
            "source-label.json",
            json.dumps(_create_label_data(), ensure_ascii=False).encode("utf-8"),
        )

    crop_label_tar(
        input_tar,
        output_tar,
        cropped_jpg_filename="frame_00000002_cropped.jpg",
        json_member_basename="frame_00000002_cropped_jpg_Label.json",
    )

    with tarfile.open(output_tar, "r") as archive:
        assert archive.getnames() == ["frame_00000002_cropped_jpg_Label.json"]
        output_label = json.load(archive.extractfile(archive.getnames()[0]))

    assert output_label["FileInfo"]["Name"] == "frame_00000002_cropped.jpg"
    assert output_label["FileName"] == "frame_00000002_cropped_jpg"
