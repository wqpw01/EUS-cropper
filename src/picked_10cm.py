"""Core operations for the fixed 10 cm image and label crop."""

import io
import json
import tarfile
import tempfile
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
from PIL import Image

from .config import CropRegion
from .label_processor import transform_label_json


CROP_BOUNDS = (603, 123, 1563, 1083)
CROP_SIZE = (960, 960)
_SOURCE_Y_MAX = 1080
_CROP_REGION = CropRegion(*CROP_BOUNDS)


def crop_image_to_canvas(image: Image.Image) -> Image.Image:
    """Crop the source image into the fixed-size RGB canvas."""
    canvas = Image.new("RGB", CROP_SIZE, "white")
    source = image.convert("RGB")
    cropped = source.crop((CROP_BOUNDS[0], CROP_BOUNDS[1], CROP_BOUNDS[2], _SOURCE_Y_MAX))
    canvas.paste(cropped, (0, 0))
    return canvas


def _crop_translation() -> np.ndarray:
    translation = np.eye(4)
    translation[0, 3] = CROP_BOUNDS[0]
    translation[1, 3] = CROP_BOUNDS[1]
    return translation


def crop_nifti_image(image: nib.Nifti1Image) -> nib.Nifti1Image:
    """Crop a two-dimensional NIfTI image while preserving its world geometry."""
    source_data = np.asanyarray(image.dataobj)
    if source_data.ndim != 2:
        raise ValueError("Expected a two-dimensional NIfTI image")
    if source_data.shape[0] < CROP_BOUNDS[2] or source_data.shape[1] < _SOURCE_Y_MAX:
        raise ValueError("NIfTI image is smaller than the fixed crop source region")

    cropped_data = np.zeros(CROP_SIZE, dtype=np.uint16)
    cropped_data[:, : _SOURCE_Y_MAX - CROP_BOUNDS[1]] = source_data[
        CROP_BOUNDS[0] : CROP_BOUNDS[2], CROP_BOUNDS[1] : _SOURCE_Y_MAX
    ]

    translation = _crop_translation()
    result = nib.Nifti1Image(
        cropped_data,
        image.affine @ translation,
        header=image.header.copy(),
    )
    result.set_data_dtype(np.uint16)

    qform, qform_code = image.get_qform(coded=True)
    if qform is not None:
        result.set_qform(qform @ translation, code=int(qform_code))

    sform, sform_code = image.get_sform(coded=True)
    if sform is not None:
        result.set_sform(sform @ translation, code=int(sform_code))

    return result


def _find_member(archive: tarfile.TarFile, suffix: str) -> Optional[tarfile.TarInfo]:
    return next(
        (
            member
            for member in archive.getmembers()
            if member.isfile() and member.name.lower().endswith(suffix)
        ),
        None,
    )


def _find_nifti_member(archive: tarfile.TarFile) -> Optional[tarfile.TarInfo]:
    return next(
        (
            member
            for member in archive.getmembers()
            if member.isfile()
            and (member.name.lower().endswith(".nii.gz") or member.name.lower().endswith(".nii"))
        ),
        None,
    )


def _read_member(archive: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    file_object = archive.extractfile(member)
    if file_object is None:
        raise ValueError(f"Unable to read TAR member: {member.name}")
    return file_object.read()


def _crop_nifti_bytes(content: bytes, suffix: str) -> bytes:
    with tempfile.TemporaryDirectory() as temporary_directory:
        source_path = Path(temporary_directory) / f"source{suffix}"
        output_path = Path(temporary_directory) / f"cropped{suffix}"
        source_path.write_bytes(content)
        source_image = nib.load(str(source_path))
        crop_nifti_image(source_image).to_filename(str(output_path))
        return output_path.read_bytes()


def _encoded_file_name(cropped_jpg_filename: str) -> str:
    cropped_path = Path(cropped_jpg_filename)
    return f"{cropped_path.stem}_{cropped_path.suffix.lstrip('.').lower()}"


def _add_tar_member(archive: tarfile.TarFile, basename: str, content: bytes) -> None:
    member = tarfile.TarInfo(Path(basename).name)
    member.size = len(content)
    archive.addfile(member, io.BytesIO(content))


def crop_label_tar(
    input_tar_path: Path,
    output_tar_path: Path,
    cropped_jpg_filename: str,
    json_member_basename: str,
    nifti_member_basename: Optional[str] = None,
) -> None:
    """Transform a label TAR and optionally crop its NIfTI mask member."""
    with tarfile.open(input_tar_path, "r:*") as input_archive:
        json_member = _find_member(input_archive, ".json")
        if json_member is None:
            raise ValueError("Input label TAR does not contain a JSON member")

        label_data = json.loads(_read_member(input_archive, json_member).decode("utf-8"))
        transformed_label = transform_label_json(label_data, _CROP_REGION)
        file_info = transformed_label.setdefault("FileInfo", {})
        file_info["Width"] = CROP_SIZE[0]
        file_info["Height"] = CROP_SIZE[1]
        file_info["Name"] = Path(cropped_jpg_filename).name
        transformed_label["FileName"] = _encoded_file_name(cropped_jpg_filename)

        nifti_member = _find_nifti_member(input_archive)
        nifti_content = (
            _crop_nifti_bytes(
                _read_member(input_archive, nifti_member),
                ".nii.gz" if nifti_member.name.lower().endswith(".nii.gz") else ".nii",
            )
            if nifti_member is not None
            else None
        )

    output_tar_path = Path(output_tar_path)
    output_tar_path.parent.mkdir(parents=True, exist_ok=True)
    json_content = json.dumps(transformed_label, ensure_ascii=False, indent=2).encode("utf-8")
    with tarfile.open(output_tar_path, "w") as output_archive:
        _add_tar_member(output_archive, json_member_basename, json_content)
        if nifti_content is not None:
            if nifti_member_basename is None:
                raise ValueError("An output NIfTI member basename is required when input includes NIfTI")
            _add_tar_member(output_archive, nifti_member_basename, nifti_content)
