# EUS 三类解剖血管特征与可视化 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在固定 10 cm EUS 裁剪流程中保留通用静脉/动脉检索，并为 IVC、Ao、门静脉系新增独立特征和三张合并边界图。

**Architecture:** `src/cropped_retrieval.py` 继续产生兼容的通用特征，同时使用独立的血管规格表从同一裁剪后 NIfTI 标签图提取三类解剖血管特征。`scripts/crop_picked_10cm.py` 从原始和裁剪后的 JSON 多边形绘制三张新增图；批处理仍通过暂存目录原子发布结果。

**Tech Stack:** Python 3、NumPy、SciPy `ndimage`、Nibabel、Pillow、pytest、mamba。

---

## 文件结构

- 修改 `src/cropped_retrieval.py`：定义三类解剖血管规格、复用连通域提取、写入 v2 检索详情字段，并校验新增裁剪图尺寸。
- 修改 `scripts/crop_picked_10cm.py`：绘制固定红/蓝/紫边界图，生成和校验三张新 PNG。
- 修改 `tests/test_cropped_retrieval.py`：用裁剪后 NIfTI 夹具锁定特征归并、来源 ID、触边剔除和通用接口兼容性。
- 修改 `tests/test_batch_picked_10cm.py`：用一帧完整输入锁定三张 PNG、精确 RGB 调色板、14 个帧内文件和 v2 详情 JSON。
- 修改 `README.md`：说明新增产物、标签归并、颜色和图库兼容性。

### Task 1: 为三类解剖血管特征建立失败测试

**Files:**
- Modify: `tests/test_cropped_retrieval.py:18-45, 287-332`
- Modify: `src/cropped_retrieval.py:18-101, 279-340, 399-520`

- [ ] **Step 1: 写入失败的三类特征测试**

在 `tests/test_cropped_retrieval.py` 中增加以下测试；它使用已有 `cropped_label_tar` 夹具，其 NIfTI 数组保持 `[x, y]` 轴顺序：

```python
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
            "source_label_ids": [29],
            "component_index": 2,
            "area_px": 1,
            "centroid_px": [0.0, 8.0],
            "reason": "touches_image_edge",
        }
    ]
    assert {item["label"] for item in details["features"]} <= {"vein", "artery"}
    assert {item["label"] for item in details["adapter_record"]["features"]} <= {
        "vein",
        "artery",
    }
```

- [ ] **Step 2: 运行测试并确认失败原因是缺少新详情字段**

Run:

```bash
mamba run -n base python -m pytest tests/test_cropped_retrieval.py::test_process_cropped_folder_writes_anatomical_vessel_features -v
```

Expected: FAIL，原因是当前详情 JSON 的 schema 仍为 v1，且不存在 `anatomical_vessel_features`。

- [ ] **Step 3: 用可复用规格表实现特征提取**

在 `src/cropped_retrieval.py` 将 `_VesselSpec` 扩展为能指定固定显示名和是否写入来源 ID 的规格，并将当前 `_VESSEL_SPECS` 重命名为 `_GENERIC_VESSEL_SPECS`。将现有规格定义替换为下列完整定义：

```python
@dataclass(frozen=True)
class _VesselSpec:
    label: str
    label_id: int
    source_label_ids: frozenset[int]
    label_desc: str | None = None
    include_source_label_ids: bool = False


_GENERIC_VESSEL_SPECS = (
    _VesselSpec(
        "vein",
        26,
        PORTAL_SOURCE_LABEL_IDS,
        include_source_label_ids=True,
    ),
    *(
        _VesselSpec("vein", label_id, frozenset({label_id}))
        for label_id in (28, 29, 30, 31, 32)
    ),
    *(
        _VesselSpec("artery", label_id, frozenset({label_id}))
        for label_id in (3, 33, 34, 35, 36, 37, 38, 39, 40)
    ),
)


_ANATOMICAL_VESSEL_SPECS = (
    _VesselSpec("aorta", 3, frozenset({3, 33}), "腹主动脉", True),
    _VesselSpec("inferior_vena_cava", 30, frozenset({30}), "下腔静脉", True),
    _VesselSpec(
        "portal_venous_system",
        26,
        frozenset({26, 27, 28, 29}),
        "门静脉系",
        True,
    ),
)
```

用下列参数化实现替代当前 `_features()` 中的循环。它对每个规格以 `np.isin()` 建立类别二值掩膜、用 8 连通 `ndimage.label()` 分割、从该连通域实际像素取得排序后的 `source_label_ids`；触边时写入跳过列表，否则写入毫米坐标和面积：

```python
def _extract_features(
    labels: np.ndarray,
    table: dict[int, dict[str, Any]],
    pixel_spacing_mm: tuple[float, float],
    specs: tuple[_VesselSpec, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    height, width = labels.shape
    x_spacing, y_spacing = pixel_spacing_mm
    structure = np.ones((3, 3), dtype=np.uint8)
    features: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for spec in specs:
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
                "label_desc": spec.label_desc
                or table.get(spec.label_id, {}).get(
                    "description", f"label_{spec.label_id}"
                ),
                "component_index": component_index,
                "area_px": int(len(points_yx)),
                "centroid_px": [float(np.mean(x_values)), float(np.mean(y_values))],
            }
            if spec.include_source_label_ids:
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
```

保留下列两个薄包装函数，使通用接口语义明确：

```python
def _features(labels, table, pixel_spacing_mm):
    return _extract_features(labels, table, pixel_spacing_mm, _GENERIC_VESSEL_SPECS)


def _anatomical_vessel_features(labels, table, pixel_spacing_mm):
    return _extract_features(
        labels, table, pixel_spacing_mm, _ANATOMICAL_VESSEL_SPECS
    )
```

通用门静脉 `26/27` 规格仍须设置 `include_source_label_ids=True`，确保现有 `features` 的门户来源审计字段不变。通用非门户特征不能新增 `source_label_ids`，以保持现有 JSON 契约。

将 `_SCHEMA_VERSION` 改为 `cropped-retrieval-features/v2`。在 `_write_artifacts()` 中新增两个特征列表和以下可视化引用；`_gallery_record()` 不改动：

```python
"anatomical_vessel_visualizations": {
    "original_overlay_png": f"{stem}_original_ivc_ao_pv_overlay.png",
    "cropped_overlay_png": f"{stem}_cropped_ivc_ao_pv_overlay.png",
    "boundary_only_png": f"{stem}_cropped_ivc_ao_pv_label_white.png",
},
"anatomical_vessel_features": anatomical_features,
"anatomical_vessel_skipped_components": anatomical_skipped,
```

在 `process_cropped_folder()` 中，先计算通用结果，再使用 `_anatomical_vessel_features()` 计算新增结果并传给 `_write_artifacts()`。向 `_validate_visual_images()` 的可选尺寸校验列表增加以下裁剪图后缀，保持独立调用时“文件不存在则跳过”的既有行为：

```python
"_cropped_ivc_ao_pv_overlay.png",
"_cropped_ivc_ao_pv_label_white.png",
```

- [ ] **Step 4: 运行新测试和现有检索测试**

Run:

```bash
mamba run -n base python -m pytest tests/test_cropped_retrieval.py -v
```

Expected: PASS，原有通用血管、器官元数据、JSON-only 标签和门静脉 `26/27` 测试全部保持通过。

- [ ] **Step 5: 提交特征元数据改动**

```bash
git add src/cropped_retrieval.py tests/test_cropped_retrieval.py
git commit -m "feat: add anatomical vessel retrieval features"
```

### Task 2: 为三类血管建立独立的边界绘制器

**Files:**
- Modify: `scripts/crop_picked_10cm.py:18-109`
- Modify: `tests/test_batch_picked_10cm.py:11-84`

- [ ] **Step 1: 写入失败的颜色绘制测试**

从 `scripts.crop_picked_10cm` 导入 `draw_anatomical_vessel_outlines`，并新增下列只使用 20 x 20 白色画布的测试。它逐一覆盖 ID `3`、`30`、`26`、`27`、`28`、`29`，并确认肝静脉 ID `31` 不会出现在新图中：

```python
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
        "Polys": [{"Shapes": [
            shape(3, 2, 3), shape(30, 7, 3), shape(26, 12, 2),
            shape(27, 12, 6), shape(28, 12, 10), shape(29, 12, 14),
            shape(31, 17, 3),
        ]}],
    }
    result = draw_anatomical_vessel_outlines(
        Image.new("RGB", (20, 20), "white"), label_data
    )
    assert result.getpixel((2, 3)) == (255, 0, 0)
    assert result.getpixel((7, 3)) == (0, 0, 255)
    assert {result.getpixel((12, y)) for y in (2, 6, 10, 14)} == {
        (170, 85, 255)
    }
    assert result.getpixel((17, 3)) == (255, 255, 255)
```

- [ ] **Step 2: 运行测试并确认导入失败**

Run:

```bash
mamba run -n base python -m pytest tests/test_batch_picked_10cm.py::test_draw_anatomical_vessel_outlines_uses_fixed_three_class_palette -v
```

Expected: FAIL，原因是 `draw_anatomical_vessel_outlines` 尚未定义。

- [ ] **Step 3: 实现固定调色板绘制器**

在 `scripts/crop_picked_10cm.py` 定义固定的 ID 到颜色映射，并用现有 `get_polygons_from_label()` 的 `points` 与 `closed` 语义绘制线条：

```python
ANATOMICAL_VESSEL_COLORS = {
    3: (255, 0, 0),
    33: (255, 0, 0),
    30: (0, 0, 255),
    26: (170, 85, 255),
    27: (170, 85, 255),
    28: (170, 85, 255),
    29: (170, 85, 255),
}


def draw_anatomical_vessel_outlines(
    image: Image.Image, label_data: dict, line_width: int = 2
) -> Image.Image:
    canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for polygon in get_polygons_from_label(label_data):
        color = ANATOMICAL_VESSEL_COLORS.get(polygon["label"])
        if color is None:
            continue
        points = [tuple(point[:2]) for point in polygon["points"] if len(point) >= 2]
        if len(points) == 1:
            x, y = points[0]
            draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=color)
        elif points:
            path = points + [points[0]] if polygon.get("closed", False) else points
            draw.line(path, fill=color, width=line_width, joint="curve")
    return canvas
```

不得修改 `draw_vessel_outlines()`、`VEIN_COLOR` 或 `ARTERY_COLOR`；它们继续生成原有通用静脉/动脉可视化。

- [ ] **Step 4: 运行颜色绘制测试**

Run:

```bash
mamba run -n base python -m pytest tests/test_batch_picked_10cm.py::test_draw_anatomical_vessel_outlines_uses_fixed_three_class_palette -v
```

Expected: PASS，红、蓝、紫各自只对应确认过的标签 ID，肝静脉 `31` 不绘制。

- [ ] **Step 5: 提交绘制器改动**

```bash
git add scripts/crop_picked_10cm.py tests/test_batch_picked_10cm.py
git commit -m "feat: draw anatomical vessel boundaries"
```

### Task 3: 在批处理中生成三张图并锁定端到端产物

**Files:**
- Modify: `scripts/crop_picked_10cm.py:151-191`
- Modify: `tests/test_batch_picked_10cm.py:20-200`

- [ ] **Step 1: 扩展批处理夹具和失败断言**

扩展 `_label_data()`，让 `ColorLabelTableModel` 和 `Polys[0].Shapes` 包含 `3`、`30`、`26`、`27`、`28`、`29` 及一个不属于新类别的 `31`。用下列固定位置保证所有轮廓均在裁剪范围 `(603, 123, 1563, 1083)` 内且不重叠：

```python
def _shape(label_id: int, x: int, y: int) -> dict:
    return {
        "labelType": label_id,
        "Points": [
            {"Pos": [x, y, 0.0]},
            {"Pos": [x + 20, y, 0.0]},
            {"Pos": [x + 20, y + 20, 0.0]},
            {"Pos": [x, y + 20, 0.0]},
        ],
    }


shapes = [
    _shape(3, 650, 180), _shape(30, 720, 180),
    _shape(26, 790, 180), _shape(27, 830, 180),
    _shape(28, 870, 180), _shape(29, 910, 180),
    _shape(31, 950, 180),
]
```

在 `_source_nifti_bytes()` 中加入同样的来源 ID：

```python
labels_xy[650:652, 180:182] = 3
labels_xy[720:722, 180:182] = 30
labels_xy[790:792, 180:182] = 26
labels_xy[830:832, 180:182] = 27
labels_xy[870:872, 180:182] = 28
labels_xy[910:912, 180:182] = 29
labels_xy[950:952, 180:182] = 31
```

将帧目录断言扩展为 14 个文件，并增加：

```python
assert "frame_00000001_original_ivc_ao_pv_overlay.png" in names
assert "frame_00000001_cropped_ivc_ao_pv_overlay.png" in names
assert "frame_00000001_cropped_ivc_ao_pv_label_white.png" in names
```

读取白底图并断言尺寸与调色板：

```python
with Image.open(frame_dir / "frame_00000001_cropped_ivc_ao_pv_label_white.png") as image:
    assert image.size == (960, 960)
    colors = {tuple(value) for value in np.unique(np.asarray(image).reshape(-1, 3), axis=0)}
    assert colors <= {
        (255, 255, 255),
        (255, 0, 0),
        (0, 0, 255),
        (170, 85, 255),
    }
    assert (255, 0, 0) in colors
    assert (0, 0, 255) in colors
    assert (170, 85, 255) in colors
```

断言原图新叠加图为 `1920 x 1080`、裁剪新叠加图为 `960 x 960`；详情 JSON 为 v2，新增图名与 `anatomical_vessel_visualizations` 一致，图库 JSONL 的 `features` 仍只有 `vein` / `artery`。

- [ ] **Step 2: 运行端到端测试并确认缺少新文件**

Run:

```bash
mamba run -n base python -m pytest tests/test_batch_picked_10cm.py::test_run_batch_writes_retrieval_artifacts_with_vessel_only_visualizations -v
```

Expected: FAIL，当前 `_process_item()` 尚未写入三个 `ivc_ao_pv` PNG。

- [ ] **Step 3: 生成并校验三张新图**

在 `_process_item()` 声明以下路径：

```python
original_anatomical_overlay_path = frame_dir / f"{image_path.stem}_original_ivc_ao_pv_overlay.png"
cropped_anatomical_overlay_path = frame_dir / f"{image_path.stem}_cropped_ivc_ao_pv_overlay.png"
cropped_anatomical_white_path = frame_dir / f"{image_path.stem}_cropped_ivc_ao_pv_label_white.png"
```

在现有 `with Image.open(image_path)` 上下文内先保留原图的新叠加对象，随后在上下文外写入三个 PNG：

```python
with Image.open(image_path) as source_image:
    original_overlay = draw_label_outlines(source_image, label_data)
    original_anatomical_overlay = draw_anatomical_vessel_outlines(
        source_image, label_data
    )
    cropped_image = crop_image_to_canvas(source_image)

original_anatomical_overlay.save(original_anatomical_overlay_path)
draw_anatomical_vessel_outlines(cropped_image, transformed_label).save(
    cropped_anatomical_overlay_path
)
draw_anatomical_vessel_outlines(
    Image.new("RGB", cropped_image.size, "white"), transformed_label
).save(cropped_anatomical_white_path)
```

增加一个只由批处理调用的校验函数，并在调用 `process_cropped_folder(frame_dir)` 前运行它：

```python
def _validate_anatomical_vessel_images(
    original_path: Path,
    cropped_overlay_path: Path,
    cropped_white_path: Path,
    original_size: tuple[int, int],
    cropped_size: tuple[int, int],
) -> None:
    for path, expected_size in (
        (original_path, original_size),
        (cropped_overlay_path, cropped_size),
        (cropped_white_path, cropped_size),
    ):
        if not path.is_file():
            raise ValueError(f"Missing anatomical vessel visualization: {path}")
        with Image.open(path) as image:
            if image.size != expected_size:
                raise ValueError(
                    "Anatomical vessel visualization size does not match expected size: "
                    f"{path}"
                )
```

调用时传入固定原图尺寸和当前裁剪图尺寸：

```python
_validate_anatomical_vessel_images(
    original_anatomical_overlay_path,
    cropped_anatomical_overlay_path,
    cropped_anatomical_white_path,
    (1920, 1080),
    cropped_image.size,
)
```

- [ ] **Step 4: 运行批处理测试集**

Run:

```bash
mamba run -n base python -m pytest tests/test_batch_picked_10cm.py -v
```

Expected: PASS，帧内文件数为 14，白底图只含四种允许颜色，详情 JSON 新字段与图库兼容字段均正确。

- [ ] **Step 5: 提交批处理图像产物改动**

```bash
git add scripts/crop_picked_10cm.py tests/test_batch_picked_10cm.py
git commit -m "feat: export combined anatomical vessel visualizations"
```

### Task 4: 更新文档并进行代码级回归验证

**Files:**
- Modify: `README.md:7-20`

- [ ] **Step 1: 更新 README 输出契约**

将“每帧 11 个文件”更新为“每帧 14 个文件”，并加入以下说明：

```markdown
检索详情 JSON v2 额外提供 `anatomical_vessel_features` 和
`anatomical_vessel_skipped_components`。它们分别记录裁剪范围内完整的
Ao、IVC、PV 连通域，以及触及裁剪边缘而不参与检索的连通域。PV 固定合并
ID `26, 27, 28, 29`，其中 `28` 为肠系膜上静脉、`29` 为脾静脉。

每帧另有三张合并边界图：原图叠加、裁剪图叠加和白底图。Ao 使用红色
`(255, 0, 0)`，IVC 使用蓝色 `(0, 0, 255)`，PV 使用紫色
`(170, 85, 255)`。原有图库 JSONL 和通用静脉/动脉图保持不变。
```

- [ ] **Step 2: 运行全部自动化测试**

Run:

```bash
mamba run -n base python -m pytest -q
```

Expected: PASS；允许已有第三方弃用警告，但不允许测试失败。

- [ ] **Step 3: 检查本次差异只覆盖计划范围**

Run:

```bash
git diff --check HEAD
git status --short
git diff --stat HEAD
```

Expected: 无空白错误；本任务尚未提交时未暂存差异只包含 `README.md`，此前三个功能提交只涉及 `src/cropped_retrieval.py`、`scripts/crop_picked_10cm.py` 和两个测试文件。

- [ ] **Step 4: 提交文档**

```bash
git add README.md
git commit -m "docs: describe anatomical vessel outputs"
```

### Task 5: 重跑桌面裁剪、验证结果并推送 GitHub

**Files:**
- Create: `/mnt/c/Users/zhangyutang/Desktop/学姐标注EUS_10cm裁剪结果/`（批处理结果，不加入 Git）

- [ ] **Step 1: 预检输入和输出目录**

Run:

```bash
find /mnt/c/Users/zhangyutang/Desktop/学姐标注EUS -maxdepth 1 -type f -iname '*.jpg' | wc -l
test ! -e /mnt/c/Users/zhangyutang/Desktop/学姐标注EUS_10cm裁剪结果
```

Expected: 输入为 105 张 JPG；第二条命令成功，确认用户已删除旧结果且不需要 `--overwrite`。

- [ ] **Step 2: 重跑固定 10 cm 裁剪**

Run:

```bash
mamba run -n base python scripts/crop_picked_10cm.py \
  --input-dir /mnt/c/Users/zhangyutang/Desktop/学姐标注EUS \
  --output-dir /mnt/c/Users/zhangyutang/Desktop/学姐标注EUS_10cm裁剪结果
```

Expected: 处理 105 帧，输出目录只在全部帧成功后出现。

- [ ] **Step 3: 验证结构、元数据和代表图像**

Run:

```bash
find /mnt/c/Users/zhangyutang/Desktop/学姐标注EUS_10cm裁剪结果 -mindepth 1 -maxdepth 1 -type d | wc -l
find /mnt/c/Users/zhangyutang/Desktop/学姐标注EUS_10cm裁剪结果 -type f | wc -l
```

Expected: 105 个帧目录、1471 个文件（`105 * 14` 个帧内文件加根目录 `eus_possible_organs.json`）。随后用 Pillow 打开至少一帧的三张新 PNG，确认原图为 `1920 x 1080`，两张裁剪图为 `960 x 960`，并在白底图中检查允许调色板和三类颜色。读取同帧详情 JSON，确认 `schema_version` 为 v2，PV 特征只使用 `26-29` 子集，图库 JSONL 的特征仍只使用 `vein` / `artery`。

- [ ] **Step 4: 提交并推送所有实现提交**

Run:

```bash
git status --short --branch
git log --oneline origin/main..HEAD
git push origin main
git status --short --branch
```

Expected: 所有计划内提交已推送到 `origin/main`，最终工作树干净且本地 `main` 与 `origin/main` 同步。用户已取消远程服务器同步，因此不执行 `rsync`、`scp` 或 SSH 服务器发布。
