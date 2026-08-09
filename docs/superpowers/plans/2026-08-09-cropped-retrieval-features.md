# Fixed 10 cm Crop Retrieval Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add automatic artery and vein retrieval-feature export to every successful fixed 10 cm EUS crop frame.

**Architecture:** Introduce `src/cropped_retrieval.py` as the single owner of cropped-TAR parsing, NIfTI-to-image coordinate conversion, connected-component extraction, and per-frame JSON/JSONL export. `scripts/crop_picked_10cm.py` calls it after the cropped TAR and vessel-only visualizations exist, preserving the existing staging-directory atomic publish behavior.

**Tech Stack:** Python 3, NumPy, nibabel, SciPy `ndimage`, Pillow, pytest.

---

## File Structure

- Create: `src/cropped_retrieval.py`
  - Parses one cropped label TAR, transforms NIfTI axes into image `[y, x]` order, extracts complete vessel connected components, and writes the two retrieval files.
- Create: `tests/test_cropped_retrieval.py`
  - Tests standalone feature extraction, label grouping, edge exclusion, coordinate conversion, and JSON-only labels.
- Modify: `scripts/crop_picked_10cm.py`
  - Invokes feature extraction after writing the cropped TAR and returns the count of gallery frames.
- Modify: `tests/test_batch_picked_10cm.py`
  - Makes the fixture contain NIfTI labels and verifies the two new artifacts and batch summary.
- Modify: `requirements.txt`
  - Declares SciPy for connected-component labeling.
- Modify: `README.md`
  - Documents the automatic files and their coordinate convention.

### Task 1: Add Standalone Retrieval Tests

**Files:**
- Create: `tests/test_cropped_retrieval.py`

- [ ] **Step 1: Write a TAR fixture helper with an asymmetric NIfTI label map**

```python
def _add_tar_member(archive: tarfile.TarFile, name: str, content: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(content)
    archive.addfile(member, io.BytesIO(content))


def _write_label_tar(directory: Path, labels_xy: np.ndarray | None) -> None:
    stem = directory.name
    metadata = {
        "FileInfo": {"Width": 10, "Height": 10},
        "Models": {"ColorLabelTableModel": [
            {"ID": 3, "Desc": "腹主动脉", "Color": [255, 0, 0, 255]},
            {"ID": 26, "Desc": "门静脉", "Color": [0, 188, 212, 255]},
            {"ID": 33, "Desc": "腹腔干", "Color": [255, 82, 0, 255]},
            {"ID": 15, "Desc": "非血管", "Color": [0, 85, 0, 255]},
        ]},
    }
    tar_path = directory / f"{stem}_cropped_jpg_Label.tar"
    with tarfile.open(tar_path, "w") as archive:
        json_payload = json.dumps(metadata, ensure_ascii=False).encode("utf-8")
        _add_tar_member(archive, f"{stem}_cropped_jpg_Label.json", json_payload)
        if labels_xy is not None:
            nifti_path = directory / "labels.nii.gz"
            nib.Nifti1Image(labels_xy, np.eye(4)).to_filename(nifti_path)
            _add_tar_member(
                archive,
                f"{stem}_cropped_jpg_Label.nii.gz",
                nifti_path.read_bytes(),
            )
```

- [ ] **Step 2: Add a failing complete-component test**

```python
def test_process_cropped_folder_extracts_complete_vessels_in_image_coordinates(tmp_path):
    from src.cropped_retrieval import process_cropped_folder

    folder = tmp_path / "frame_00000001"
    folder.mkdir()
    labels_xy = np.zeros((10, 10), dtype=np.uint16)
    labels_xy[2:4, 5:7] = 26       # x = 2.5, y = 5.5 after transpose.
    labels_xy[6:8, 2:4] = 3        # ID 3 must be an artery.
    labels_xy[0:2, 4:6] = 33       # Touches x=0 and must be skipped.
    labels_xy[4:6, 7:9] = 15       # Not a vessel.
    _write_label_tar(folder, labels_xy)

    result = process_cropped_folder(folder)

    details = json.loads(result.feature_path.read_text(encoding="utf-8"))
    assert [(item["label"], item["label_id"]) for item in details["features"]] == [
        ("vein", 26), ("artery", 3)
    ]
    assert details["features"][0]["centroid_px"] == [2.5, 5.5]
    assert details["features"][0]["x_mm"] == 2.5 * 100.0 / 9.0
    assert details["features"][0]["y_mm"] == 5.5 * 100.0 / 9.0
    assert details["skipped_components"] == [{
        "label": "artery",
        "label_id": 33,
        "label_desc": "腹腔干",
        "component_index": 1,
        "area_px": 4,
        "centroid_px": [0.5, 4.5],
        "reason": "touches_image_edge",
    }]
```

- [ ] **Step 3: Add a failing JSON-only label test**

```python
def test_process_cropped_folder_writes_unindexed_record_for_json_only_tar(tmp_path):
    from src.cropped_retrieval import process_cropped_folder

    folder = tmp_path / "frame_00000002"
    folder.mkdir()
    _write_label_tar(folder, None)

    result = process_cropped_folder(folder)

    details = json.loads(result.feature_path.read_text(encoding="utf-8"))
    record = json.loads(result.gallery_path.read_text(encoding="utf-8"))
    assert details["label_source"] == "empty_label_json"
    assert details["features"] == []
    assert details["skipped_components"] == []
    assert record["status"] == "unindexed"
    assert record["features"] == []
```

- [ ] **Step 4: Run the test file to verify the expected import failure**

Run: `pytest tests/test_cropped_retrieval.py -q`

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'src.cropped_retrieval'`.

- [ ] **Step 5: Commit the failing tests**

```bash
git add tests/test_cropped_retrieval.py
git commit -m "test: define cropped retrieval feature behavior"
```

### Task 2: Implement the Cropped Retrieval Module

**Files:**
- Create: `src/cropped_retrieval.py`
- Modify: `requirements.txt`
- Test: `tests/test_cropped_retrieval.py`

- [ ] **Step 1: Define the module contract and label groups**

```python
VEIN_LABEL_IDS = frozenset({26, 27, 28, 29, 30, 31, 32})
ARTERY_LABEL_IDS = frozenset({3, 33, 34, 35, 36, 37, 38, 39, 40})
_LABEL_GROUPS = (("vein", VEIN_LABEL_IDS), ("artery", ARTERY_LABEL_IDS))
_SCHEMA_VERSION = "cropped-retrieval-features/v1"

@dataclass(frozen=True)
class CroppedFeatureResult:
    folder: Path
    feature_path: Path
    gallery_path: Path
    record: dict[str, Any]

def process_cropped_folder(
    folder: str | Path, width_mm: float = 100.0, length_mm: float = 100.0
) -> CroppedFeatureResult:
    directory = Path(folder)
    stem = directory.name
    if width_mm <= 0.0 or length_mm <= 0.0:
        raise ValueError("Crop physical dimensions must be positive")
    metadata, payload, suffix = _tar_members(
        directory / f"{stem}_cropped_jpg_Label.tar"
    )
    if payload is None:
        labels = _empty_label_image(metadata)
        label_source = "empty_label_json"
    else:
        if suffix is None:
            raise ValueError("NIfTI payload is missing its filename suffix")
        labels = _read_label_image(payload, suffix)
        label_source = "nifti"
    expected_width = int(metadata["FileInfo"]["Width"])
    expected_height = int(metadata["FileInfo"]["Height"])
    if labels.shape != (expected_height, expected_width):
        raise ValueError(
            f"Label image size {labels.shape[::-1]} does not match JSON "
            f"size {(expected_width, expected_height)}"
        )
    table = _label_table(metadata)
    height, width = labels.shape
    spacing = (width_mm / (width - 1), length_mm / (height - 1))
    features, skipped = _features(labels, table, spacing)
    record = _gallery_record(stem, features, width_mm, length_mm, spacing)
    return _write_artifacts(
        directory, stem, labels, label_source, features, skipped,
        width_mm, length_mm, spacing, record,
    )
```

- [ ] **Step 2: Implement strict TAR parsing and NIfTI coordinate conversion**

```python
def _tar_members(tar_path: Path) -> tuple[dict[str, Any], bytes | None, str | None]:
    with tarfile.open(tar_path, "r:*") as archive:
        json_members = [
            member for member in archive.getmembers()
            if member.isfile() and member.name.lower().endswith(".json")
        ]
        nifti_members = [
            member for member in archive.getmembers()
            if member.isfile()
            and (member.name.lower().endswith(".nii.gz") or member.name.lower().endswith(".nii"))
        ]
        if len(json_members) != 1 or len(nifti_members) > 1:
            raise ValueError(f"Label TAR must contain one JSON and at most one NIfTI: {tar_path}")
        json_stream = archive.extractfile(json_members[0])
        if json_stream is None:
            raise ValueError(f"Unable to read label JSON: {tar_path}")
        metadata = json.load(json_stream)
        if not nifti_members:
            return metadata, None, None
        nifti_stream = archive.extractfile(nifti_members[0])
        if nifti_stream is None:
            raise ValueError(f"Unable to read label NIfTI: {tar_path}")
        suffix = ".nii.gz" if nifti_members[0].name.lower().endswith(".nii.gz") else ".nii"
        return metadata, nifti_stream.read(), suffix


def _read_label_image(payload: bytes, suffix: str) -> np.ndarray:
    with tempfile.TemporaryDirectory() as temporary_directory:
        path = Path(temporary_directory) / f"label{suffix}"
        path.write_bytes(payload)
        values_xy = np.asanyarray(nib.load(str(path)).dataobj)
    if values_xy.ndim != 2:
        raise ValueError(f"Cropped label NIfTI must be two-dimensional: {values_xy.shape}")
    return np.asarray(values_xy).T

def _empty_label_image(metadata: dict[str, Any]) -> np.ndarray:
    width = int(metadata["FileInfo"]["Width"])
    height = int(metadata["FileInfo"]["Height"])
    if width < 2 or height < 2:
        raise ValueError(f"Invalid JSON-only label size: {width} x {height}")
    return np.zeros((height, width), dtype=np.uint16)


def _label_table(metadata: dict[str, Any]) -> dict[int, dict[str, Any]]:
    entries = metadata["Models"]["ColorLabelTableModel"]
    if not isinstance(entries, list):
        raise ValueError("ColorLabelTableModel must be a list")
    result: dict[int, dict[str, Any]] = {}
    for entry in entries:
        if isinstance(entry, dict) and isinstance(entry.get("ID"), int):
            result[entry["ID"]] = {"description": str(entry.get("Desc", f"label_{entry['ID']}"))}
    return result
```

`_tar_members()` rejects zero or multiple JSON members and more than one NIfTI member. It returns the decoded metadata, optional NIfTI bytes, and optional suffix so compressed and uncompressed NIfTI files are handled correctly.

- [ ] **Step 3: Implement 8-connected vessel feature extraction**

```python
def _features(
    labels: np.ndarray,
    table: dict[int, dict[str, Any]],
    pixel_spacing_mm: tuple[float, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    height, width = labels.shape
    x_spacing, y_spacing = pixel_spacing_mm
    structure = np.ones((3, 3), dtype=np.uint8)
    features: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for feature_label, label_ids in _LABEL_GROUPS:
        for label_id in sorted(label_ids):
            components, count = ndimage.label(labels == label_id, structure=structure)
            for component_id in range(1, count + 1):
                points_yx = np.argwhere(components == component_id)
                y_values = points_yx[:, 0]
                x_values = points_yx[:, 1]
                base = {
                    "label": feature_label,
                    "label_id": label_id,
                    "label_desc": table.get(label_id, {}).get("description", f"label_{label_id}"),
                    "component_index": component_id,
                    "area_px": int(len(points_yx)),
                    "centroid_px": [float(np.mean(x_values)), float(np.mean(y_values))],
                }
                touches_edge = bool(
                    np.any(x_values == 0) or np.any(x_values == width - 1)
                    or np.any(y_values == 0) or np.any(y_values == height - 1)
                )
                if touches_edge:
                    skipped.append({**base, "reason": "touches_image_edge"})
                else:
                    features.append({
                        **base,
                        "x_mm": float(np.mean(x_values) * x_spacing),
                        "y_mm": float(np.mean(y_values) * y_spacing),
                        "area_mm2": float(len(points_yx) * x_spacing * y_spacing),
                    })
    return features, skipped
```

Use `x_values = points_yx[:, 1]` and `y_values = points_yx[:, 0]`; this makes the module's coordinates agree with the cropped JPG and with the reference SimpleITK implementation. Compute pixel spacing as `(width_mm / (width - 1), length_mm / (height - 1))`.

- [ ] **Step 4: Write the JSON and JSONL artifacts with compatible visual paths**

```python
def _gallery_record(
    stem: str,
    features: list[dict[str, Any]],
    width_mm: float,
    length_mm: float,
    pixel_spacing_mm: tuple[float, float],
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
        "source": "cropped_label_tar",
        "probe_point_world": center,
        "input_normal_world": [0.0, 0.0, 1.0],
        "input_direction_world": [0.0, 1.0, 0.0],
        "square_vertices_world": [[0.0, 0.0, 0.0], [width_mm, 0.0, 0.0], [width_mm, length_mm, 0.0], [0.0, length_mm, 0.0]],
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
        "quality": {"accepted": True, "reason": None, "black_ratio": None, "line_length_px": None, "black_side_ratio": None, "valid_side_black_ratio": None},
        "resampling_backend": "label_tar_2d",
        "pose_coordinate_system": "synthetic_2d_10cm_crop",
        "patient_world_pose": False,
    }


def _write_artifacts(
    directory: Path, stem: str, labels: np.ndarray, label_source: str,
    features: list[dict[str, Any]], skipped: list[dict[str, Any]],
    width_mm: float, length_mm: float, spacing: tuple[float, float], record: dict[str, Any],
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
        "pixel_spacing_mm": list(spacing),
        "feature_coordinate_system": "top_left_origin_x_right_y_down_mm",
        "features": features,
        "skipped_components": skipped,
        "adapter_record": record,
    }
    feature_path.write_text(json.dumps(details, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    gallery_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    return CroppedFeatureResult(directory, feature_path, gallery_path, record)
```

Before calling `_write_artifacts()`, validate the existing vessel-only visual files only when present: open each with Pillow and require `(width, height)` equal to `labels.shape[::-1]`; raise `ValueError` on a mismatch. Write `<stem>_cropped_retrieval_features.json` with newline and indent `2`, and `<stem>_cropped_gallery.jsonl` as exactly one compact JSON line plus newline. The details JSON identifies `label_source` as `"nifti"` or `"empty_label_json"`, includes the skipped components, and embeds `adapter_record`.

- [ ] **Step 5: Declare SciPy and run module tests**

Append this exact dependency to `requirements.txt`:

```text
scipy>=1.11
```

Run: `pytest tests/test_cropped_retrieval.py -q`

Expected: PASS with both standalone extraction tests passing.

- [ ] **Step 6: Commit module and dependency**

```bash
git add src/cropped_retrieval.py requirements.txt tests/test_cropped_retrieval.py
git commit -m "feat: add cropped vessel retrieval features"
```

### Task 3: Integrate Retrieval Export into the Fixed 10 cm Batch

**Files:**
- Modify: `scripts/crop_picked_10cm.py`
- Modify: `tests/test_batch_picked_10cm.py`
- Test: `tests/test_batch_picked_10cm.py`

- [ ] **Step 1: Extend the batch fixture with a cropped NIfTI label map**

```python
import nibabel as nib

def _source_nifti_bytes(tmp_path: Path) -> bytes:
    labels_xy = np.zeros((1920, 1080), dtype=np.uint16)
    labels_xy[700:702, 200:202] = 26
    labels_xy[800:802, 300:302] = 3
    path = tmp_path / "source_labels.nii.gz"
    nib.Nifti1Image(labels_xy, np.eye(4)).to_filename(path)
    return path.read_bytes()

# Add this member in the existing fixture TAR write block:
_add_tar_member(archive, "frame_00000001_jpg_Label.nii.gz", _source_nifti_bytes(tmp_path))
```

The two source components lie within the 10 cm crop and do not touch its boundary, so the expected `gallery_records` value is `1`.

- [ ] **Step 2: Change the existing end-to-end assertions before implementation**

```python
assert result == {"total": 1, "processed": 1, "gallery_records": 1}
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
details = json.loads(
    (frame_dir / "frame_00000001_cropped_retrieval_features.json").read_text(encoding="utf-8")
)
assert [feature["label_id"] for feature in details["features"]] == [26, 3]
record = json.loads((frame_dir / "frame_00000001_cropped_gallery.jsonl").read_text(encoding="utf-8"))
assert record["boundary_only_png"] == "frame_00000001_cropped_vessel_label_white.png"
assert record["ct_overlay_png"] == "frame_00000001_cropped_vessel_overlay.png"
assert not (output_dir / "retrieval_gallery.jsonl").exists()
assert not (output_dir / "retrieval_feature_summary.json").exists()
```

Change the existing cropped-TAR assertion to:

```python
with tarfile.open(frame_dir / "frame_00000001_cropped_jpg_Label.tar", "r") as archive:
    assert archive.getnames() == [
        "frame_00000001_cropped_jpg_Label.json",
        "frame_00000001_cropped_jpg_Label.nii.gz",
    ]
```

- [ ] **Step 3: Run the batch test and verify it fails for the missing artifacts**

Run: `pytest tests/test_batch_picked_10cm.py -q`

Expected: FAIL because `run_batch()` returns no `gallery_records` and the two retrieval files do not exist.

- [ ] **Step 4: Integrate the module after `crop_label_tar()`**

```python
from src.cropped_retrieval import process_cropped_folder

# Change the existing function signature at scripts/crop_picked_10cm.py:142 to:
def _process_item(image_path: Path, label_path: Path, output_root: Path) -> bool:
    # Keep lines 143-187 unchanged through the crop_label_tar() call.
    retrieval = process_cropped_folder(frame_dir)
    return bool(retrieval.record["features"])

gallery_records = 0
for image_path, label_path in tqdm(items, desc="Processing frames"):
    gallery_records += int(_process_item(image_path, label_path, staging_dir))
return {"total": len(items), "processed": len(items), "gallery_records": gallery_records}
```

The call must occur after cropped TAR creation, so feature extraction never reads a partially written archive. No `--overwrite` option is added and the existing `FileExistsError` behavior remains unchanged.

- [ ] **Step 5: Run the focused tests**

Run: `pytest tests/test_cropped_retrieval.py tests/test_batch_picked_10cm.py -q`

Expected: PASS.

- [ ] **Step 6: Commit batch integration**

```bash
git add scripts/crop_picked_10cm.py tests/test_batch_picked_10cm.py
git commit -m "feat: export retrieval records during 10cm crop"
```

### Task 4: Document and Verify the Full Contract

**Files:**
- Modify: `README.md`
- Test: `tests/test_cropped_retrieval.py`
- Test: `tests/test_batch_picked_10cm.py`

- [ ] **Step 1: Add concise README usage and output documentation**

```markdown
## 固定 10 cm 裁剪检索特征

`scripts/crop_picked_10cm.py` 会为每个帧自动写入
`*_cropped_retrieval_features.json` 和 `*_cropped_gallery.jsonl`。
特征仅包含未接触裁剪边缘的动静脉连通域；静脉为 ID 26-32，动脉为 ID 3、33-40。
坐标以裁剪图左上角为原点，x 向右、y 向下，范围为 100 mm x 100 mm。
```

- [ ] **Step 2: Run formatting and the full test suite**

Run: `git diff --check && pytest -q`

Expected: `git diff --check` has no output and every test passes.

- [ ] **Step 3: Inspect the produced retrieval artifacts from the batch fixture**

Run: `pytest tests/test_batch_picked_10cm.py -q`

Expected: PASS; the fixture proves every generated frame has eleven files, uses the vessel-only visual paths, and produces one gallery record.

- [ ] **Step 4: Commit documentation**

```bash
git add README.md
git commit -m "docs: describe cropped retrieval outputs"
```

- [ ] **Step 5: Verify repository state and publish**

Run: `git status --short --branch && git log -3 --oneline && git push origin main`

Expected: `main` is clean, the feature commits are present, and the remote `main` has the same HEAD commit.
