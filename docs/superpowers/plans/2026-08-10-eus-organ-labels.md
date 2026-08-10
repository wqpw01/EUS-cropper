# EUS Organ Labels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Export normalized EUS organ labels from FrameLabel declarations and cropped NIfTI pixels, normalize portal-confluence ID 27 to portal ID 26, and create a possible-organ catalog in every fixed 10 cm result root.

**Architecture:** src/cropped_retrieval.py owns the fixed mapping, organ metadata, merged portal feature mask, and catalog writer. scripts/crop_picked_10cm.py writes the catalog inside its staging directory before atomic publication. Existing images and the per-frame count of eleven files do not change.

**Tech Stack:** Python 3, NumPy, nibabel, SciPy ndimage, Pillow, pytest.

---

## File Structure

- Modify: src/cropped_retrieval.py
  - Fixed EUS mapping, source union, vessel normalization, serializers, and catalog writer.
- Modify: scripts/crop_picked_10cm.py
  - Call the catalog writer after all frame processing succeeds.
- Modify: tests/test_cropped_retrieval.py
  - Cover union, exclusions, JSON-only fallback, and portal merge.
- Modify: tests/test_batch_picked_10cm.py
  - Cover root catalog creation in the real batch flow.
- Modify: README.md
  - Explain sources, dual roles, and catalog output.

## Task 1: Define Failing Per-Frame Metadata Tests

**Files:**
- Modify: tests/test_cropped_retrieval.py

- [ ] **Step 1: Parameterize fixture metadata with FrameLabel values.**

Replace the current label metadata helper with the following function and pass frame_label_ids into it from the existing nested TAR writer.

~~~python
def _label_data(frame_label_ids: list[int] | None = None) -> dict:
    return {
        "FileInfo": {"Width": 10, "Height": 10},
        "Models": {
            "ColorLabelTableModel": [
                {"ID": 2, "Desc": "S2 肝脏"},
                {"ID": 3, "Desc": "腹主动脉"},
                {"ID": 12, "Desc": "胆囊"},
                {"ID": 15, "Desc": "胆囊"},
                {"ID": 16, "Desc": "肝内胆管"},
                {"ID": 18, "Desc": "脾脏"},
                {"ID": 19, "Desc": "胰腺"},
                {"ID": 26, "Desc": "门静脉"},
                {"ID": 27, "Desc": "门静脉汇合部"},
                {"ID": 28, "Desc": "肠系膜上静脉"},
                {"ID": 30, "Desc": "下腔静脉"},
                {"ID": 33, "Desc": "腹主动脉"},
                {"ID": 40, "Desc": "胃十二指肠动脉"},
            ],
            "FrameLabelModel": {
                "FrameLabel": [
                    {"FrameCount": 0, "ItemType": 0, "Label": label_id, "ViewType": 3}
                    for label_id in frame_label_ids or []
                ]
            },
        },
    }
~~~

Change the nested fixture signature to the following and change its JSON serialization to json.dumps(_label_data(frame_label_ids)).

~~~python
def write(
    labels_xy: np.ndarray | None = None,
    frame_label_ids: list[int] | None = None,
) -> Path:
~~~

- [ ] **Step 2: Add a failing union and exclusion test.**

~~~python
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
~~~

- [ ] **Step 3: Add a failing portal normalization test.**

~~~python
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
~~~

- [ ] **Step 4: Extend the JSON-only fixture test.**

Build the JSON-only TAR with frame_label_ids equal to [19, 15, 16, 28], then assert:

~~~python
assert details["frame_label_organ_ids"] == [19]
assert details["cropped_nifti_organ_ids"] == []
assert details["organ_labels"] == ["pancreas"]
assert gallery_record["organ_labels"] == ["pancreas"]
~~~

In the existing complete-component test, replace the blanket feature-key assertion with a portal-specific assertion so the new audit field is part of the contract:

~~~python
assert set(features[0]) == {
    "label", "label_id", "label_desc", "component_index", "area_px",
    "centroid_px", "x_mm", "y_mm", "area_mm2", "source_label_ids",
}
assert features[0]["source_label_ids"] == [26]
assert set(features[1]) == {
    "label", "label_id", "label_desc", "component_index", "area_px",
    "centroid_px", "x_mm", "y_mm", "area_mm2",
}
~~~

- [ ] **Step 5: Run focused tests before code changes.**

Run: pytest tests/test_cropped_retrieval.py -q

Expected: FAIL because organ metadata and source_label_ids have not been implemented.

- [ ] **Step 6: Commit the failing test contract.**

~~~bash
git add tests/test_cropped_retrieval.py
git commit -m "test: define EUS organ metadata behavior"
~~~

## Task 2: Implement Organ Sources and Portal Merge

**Files:**
- Modify: src/cropped_retrieval.py
- Test: tests/test_cropped_retrieval.py

- [ ] **Step 1: Add fixed maps below the existing public vessel ID sets.**

~~~python
ORGAN_LABEL_BY_ID = {
    1: "liver", 2: "liver", 3: "aorta", 4: "pancreas", 5: "pancreas",
    6: "spleen", 7: "pancreas", 8: "liver", 9: "pancreas", 10: "duodenum",
    11: "pancreas", 14: "liver", 18: "spleen", 19: "pancreas", 20: "pancreas",
    21: "duodenum", 22: "adrenal_gland_left", 23: "adrenal_gland_right",
    24: "kidney_left", 25: "kidney_right", 26: "portal_vein",
    27: "portal_vein", 30: "inferior_vena_cava", 33: "aorta", 41: "duodenum",
}
PORTAL_SOURCE_LABEL_IDS = frozenset({26, 27})


@dataclass(frozen=True)
class _VesselSpec:
    label: str
    label_id: int
    source_label_ids: frozenset[int]


_VESSEL_SPECS = (
    _VesselSpec("vein", 26, PORTAL_SOURCE_LABEL_IDS),
    *(_VesselSpec("vein", label_id, frozenset({label_id})) for label_id in (28, 29, 30, 31, 32)),
    *(_VesselSpec("artery", label_id, frozenset({label_id})) for label_id in (3, 33, 34, 35, 36, 37, 38, 39, 40)),
)
~~~

Keep VEIN_LABEL_IDS and ARTERY_LABEL_IDS unchanged because existing visualization code imports their full raw taxonomy.

- [ ] **Step 2: Add explicit source extraction helpers after _label_table.**

~~~python
def _organ_ids_from_frame_label(metadata: dict[str, Any]) -> list[int]:
    models = metadata.get("Models")
    frame_model = models.get("FrameLabelModel") if isinstance(models, dict) else None
    entries = frame_model.get("FrameLabel") if isinstance(frame_model, dict) else None
    if not isinstance(entries, list):
        return []
    return sorted({
        entry["Label"]
        for entry in entries
        if isinstance(entry, dict)
        and isinstance(entry.get("Label"), int)
        and not isinstance(entry.get("Label"), bool)
        and entry["Label"] in ORGAN_LABEL_BY_ID
    })


def _organ_ids_from_cropped_labels(labels: np.ndarray) -> list[int]:
    return sorted({
        int(label_id)
        for label_id in np.unique(labels)
        if int(label_id) in ORGAN_LABEL_BY_ID
    })


def _organ_metadata(metadata: dict[str, Any], labels: np.ndarray) -> dict[str, Any]:
    frame_ids = _organ_ids_from_frame_label(metadata)
    nifti_ids = _organ_ids_from_cropped_labels(labels)
    return {
        "organ_label_source": "frame_label_and_cropped_nifti",
        "frame_label_organ_ids": frame_ids,
        "cropped_nifti_organ_ids": nifti_ids,
        "organ_labels": sorted({
            ORGAN_LABEL_BY_ID[label_id] for label_id in (*frame_ids, *nifti_ids)
        }),
    }
~~~

- [ ] **Step 3: Make _features iterate _VESSEL_SPECS.**

For each spec, form the component mask with np.isin(labels, tuple(spec.source_label_ids)), retain the existing 8-connectivity and physical-coordinate logic, and use spec.label plus spec.label_id in the record. Immediately after computing y_values and x_values, add:

~~~python
source_label_ids = sorted({
    int(label_id)
    for label_id in np.unique(labels[y_values, x_values])
    if int(label_id) in spec.source_label_ids
})
if spec.source_label_ids == PORTAL_SOURCE_LABEL_IDS:
    base["source_label_ids"] = source_label_ids
~~~

This makes connected raw 26 and 27 pixels one canonical label_id 26 component. Other vessel records retain their current fields and behavior.

- [ ] **Step 4: Add organ fields to both serializers.**

Change _gallery_record to accept organ_metadata and add:

~~~python
"organ": "unknown",
"organ_label_source": organ_metadata["organ_label_source"],
"organ_labels": organ_metadata["organ_labels"],
~~~

Change _write_artifacts to accept organ_metadata and add:

~~~python
"organ_label_source": organ_metadata["organ_label_source"],
"frame_label_organ_ids": organ_metadata["frame_label_organ_ids"],
"cropped_nifti_organ_ids": organ_metadata["cropped_nifti_organ_ids"],
"organ_labels": organ_metadata["organ_labels"],
~~~

After validating labels in process_cropped_folder, run the following before feature extraction and pass organ_metadata to both serializers.

~~~python
organ_metadata = _organ_metadata(metadata, labels)
~~~

- [ ] **Step 5: Run unit tests.**

Run: pytest tests/test_cropped_retrieval.py -q

Expected: PASS.

- [ ] **Step 6: Commit implementation.**

~~~bash
git add src/cropped_retrieval.py tests/test_cropped_retrieval.py
git commit -m "feat: add EUS organ metadata to crops"
~~~

## Task 3: Add the Fixed Possible-Organ Catalog

**Files:**
- Modify: src/cropped_retrieval.py
- Modify: scripts/crop_picked_10cm.py
- Modify: tests/test_batch_picked_10cm.py

- [ ] **Step 1: Define source names and catalog writer.**

Add this name map for every ID in ORGAN_LABEL_BY_ID:

~~~python
_EUS_LABEL_NAMES = {
    1: "S1 第二肝门", 2: "S2 肝脏", 3: "S3 腹主动脉",
    4: "S4 胰体", 5: "S5 胰尾", 6: "S6 脾门", 7: "S7 胰颈",
    8: "S8 第一肝门", 9: "S9 胰头", 10: "S10 壶腹部", 11: "S11 钩突",
    14: "肝脏", 18: "脾脏", 19: "胰腺", 20: "胰管", 21: "壶腹部",
    22: "左侧肾上腺", 23: "右侧肾上腺", 24: "左侧肾脏", 25: "右侧肾脏",
    26: "门静脉（包括分支", 27: "门静脉汇合部", 30: "下腔静脉",
    33: "腹主动脉", 41: "十二指肠肠腔",
}
~~~

Add these exact functions:

~~~python
def possible_organs_catalog() -> dict[str, Any]:
    dual_roles = {
        "aorta": ("organ_and_vessel", "artery", None),
        "inferior_vena_cava": ("organ_and_vessel", "vein", None),
        "portal_vein": ("organ_and_vessel", "vein", 26),
    }
    organs = []
    for organ_label in sorted(set(ORGAN_LABEL_BY_ID.values())):
        label_ids = sorted(
            label_id
            for label_id, mapped_label in ORGAN_LABEL_BY_ID.items()
            if mapped_label == organ_label
        )
        role, vessel_type, canonical_id = dual_roles.get(
            organ_label, ("organ", None, None)
        )
        organs.append({
            "organ_label": organ_label,
            "eus_label_ids": label_ids,
            "eus_label_names": [_EUS_LABEL_NAMES[label_id] for label_id in label_ids],
            "role": role,
            "vessel_type": vessel_type,
            "canonical_vessel_label_id": canonical_id,
        })
    return {"schema_version": "eus-possible-organs/v1", "organs": organs}


def write_possible_organs_catalog(directory: str | Path) -> Path:
    path = Path(directory) / "eus_possible_organs.json"
    path.write_text(
        json.dumps(possible_organs_catalog(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path
~~~

- [ ] **Step 2: Write the catalog in the staging root.**

Import write_possible_organs_catalog in scripts/crop_picked_10cm.py. After the frame loop finishes and immediately before staging_dir.replace(output_dir), add:

~~~python
write_possible_organs_catalog(staging_dir)
~~~

- [ ] **Step 3: Extend batch fixture and test catalog output.**

Add the following declaration to the source fixture and add a complete liver entry with ID 14 plus a spleen entry with ID 18 to its ColorLabelTableModel:

~~~python
"FrameLabelModel": {
    "FrameLabel": [
        {"FrameCount": 0, "ItemType": 0, "Label": 14, "ViewType": 3},
        {"FrameCount": 0, "ItemType": 0, "Label": 15, "ViewType": 3},
    ]
},
~~~

In _source_nifti_bytes(), place the NIfTI-only spleen label in the source crop range:

~~~python
labels_xy[900:902, 250:252] = 18
~~~

Then assert:

~~~python
assert details["frame_label_organ_ids"] == [14]
assert details["cropped_nifti_organ_ids"] == [18]
assert details["organ_labels"] == ["liver", "spleen"]

catalog = json.loads((output_dir / "eus_possible_organs.json").read_text(encoding="utf-8"))
assert catalog["schema_version"] == "eus-possible-organs/v1"
assert len(catalog["organs"]) == 11
by_name = {item["organ_label"]: item for item in catalog["organs"]}
assert by_name["aorta"]["eus_label_ids"] == [3, 33]
assert by_name["aorta"]["role"] == "organ_and_vessel"
assert by_name["aorta"]["vessel_type"] == "artery"
assert by_name["portal_vein"]["eus_label_ids"] == [26, 27]
assert by_name["portal_vein"]["canonical_vessel_label_id"] == 26
assert by_name["liver"]["role"] == "organ"
~~~

- [ ] **Step 4: Run batch integration test.**

Run: pytest tests/test_batch_picked_10cm.py -q

Expected: PASS. The frame directory retains eleven files and the batch root gains one eus_possible_organs.json file.

- [ ] **Step 5: Commit batch integration.**

~~~bash
git add src/cropped_retrieval.py scripts/crop_picked_10cm.py tests/test_batch_picked_10cm.py
git commit -m "feat: export EUS possible organ catalog"
~~~

## Task 4: Document, Verify, and Rebuild

**Files:**
- Modify: README.md
- Generated: C:\Users\zhangyutang\Desktop\学姐标注EUS_10cm裁剪结果

- [ ] **Step 1: Add this documentation paragraph.**

~~~markdown
每帧检索元数据的 organ_labels 是 JSON FrameLabelModel.FrameLabel 与裁剪后 NIfTI 实际标签的并集。胆囊、胆管和普通血管不进入器官集合；腹主动脉、下腔静脉和门静脉同时保留器官与血管语义，门静脉汇合部 ID 27 归并为 ID 26。批处理根目录同时生成 eus_possible_organs.json，固定列出全部 11 个可出现器官及其 EUS ID、中文名称和血管双重角色。
~~~

- [ ] **Step 2: Run the full test suite.**

Run: pytest -q

Expected: PASS. Existing third-party deprecation warnings are acceptable.

- [ ] **Step 3: Commit documentation.**

~~~bash
git add README.md
git commit -m "docs: describe EUS organ catalog"
~~~

- [ ] **Step 4: Confirm the output directory is absent.**

Run: test ! -e '/mnt/c/Users/zhangyutang/Desktop/学姐标注EUS_10cm裁剪结果'

Expected: exit code 0. If it exists, stop instead of deleting user data.

- [ ] **Step 5: Run the complete fixed 10 cm batch.**

Run:

~~~bash
python scripts/crop_picked_10cm.py \
  --input-dir '/mnt/c/Users/zhangyutang/Desktop/学姐标注EUS' \
  --output-dir '/mnt/c/Users/zhangyutang/Desktop/学姐标注EUS_10cm裁剪结果'
~~~

Expected: Processed 105 of 105 frames, followed by a gallery count and output path.

- [ ] **Step 6: Audit every generated frame and catalog.**

Run:

~~~bash
python -c 'import json; from pathlib import Path; root=Path("/mnt/c/Users/zhangyutang/Desktop/学姐标注EUS_10cm裁剪结果"); frames=sorted(p for p in root.iterdir() if p.is_dir()); assert len(frames)==105; catalog=json.loads((root/"eus_possible_organs.json").read_text(encoding="utf-8")); assert len(catalog["organs"])==11; required={"organ_label_source","frame_label_organ_ids","cropped_nifti_organ_ids","organ_labels"}; assert all(required <= set(json.loads((frame/f"{frame.name}_cropped_retrieval_features.json").read_text(encoding="utf-8"))) for frame in frames); print("validated", len(frames), "frames and", len(catalog["organs"]), "catalog entries")'
~~~

Expected: validated 105 frames and 11 catalog entries.

- [ ] **Step 7: Push all implementation commits.**

~~~bash
git push origin main
git status --short --branch
~~~

Expected final status: ## main...origin/main.
