# EUS 裁剪器官标签提取设计

## 目标

固定 10 cm EUS 裁剪流程在每个帧目录的检索详情 JSON 和图库 JSONL 中写入可靠的 `organ_labels`。器官集合必须同时反映：

1. 原始标签 JSON 的 `Models.FrameLabelModel.FrameLabel` 声明。
2. 裁剪后标签 TAR 中 NIfTI 掩膜实际出现的标签值。

两路取并集；不再以 `Polys` 或 `PolygonModel2` 的人工轮廓作为器官出现与否的唯一依据。这样，JSON 已声明但未勾画的器官不会遗漏，裁剪后 NIfTI 中实际出现但 JSON 未声明的器官也不会遗漏。

本功能只接入 `scripts/crop_picked_10cm.py`。既有裁剪图片、标签 TAR、血管可视化、边缘血管剔除和检索坐标规则保持不变。

批处理完成后，还要在裁剪结果根目录生成一份固定的 `eus_possible_organs.json`。它列出 EUS 中所有可能出现的规范器官，而非只列出本次输入数据实际出现的器官。

## 方案选择

采用规范器官标签数组，而不是直接输出原始中文名称或仅输出原始 ID。

- 规范名称可直接供后续 EUS--CT 检索筛选使用。
- 原始 ID 仍作为来源审计信息写入详情 JSON，因此映射可复核。
- `FrameLabel` 和裁剪后 NIfTI 分别保留来源 ID；最终 `organ_labels` 去重排序。

不采用仅依赖人工多边形的旧方式，也不采用整帧原始 NIfTI；用户已确认 NIfTI 只计算裁剪后的 10 cm 范围。

## 固定器官映射

以下 EUS 标签被视为器官，并规范化为对应输出名称：

| 规范名称 | EUS ID |
| --- | --- |
| `liver` | `1, 2, 8, 14` |
| `pancreas` | `4, 5, 7, 9, 11, 19, 20` |
| `spleen` | `6, 18` |
| `duodenum` | `10, 21, 41` |
| `adrenal_gland_left` | `22` |
| `adrenal_gland_right` | `23` |
| `kidney_left` | `24` |
| `kidney_right` | `25` |
| `aorta` | `3, 33` |
| `inferior_vena_cava` | `30` |
| `portal_vein` | `26, 27` |

以下标签不进入 `organ_labels`：胆囊 `12, 15`，胆管 `16, 17`，`S0` 的 `13`，以及其他血管 `28, 29, 31, 32, 34--40`。未知 ID 也忽略，不产生错误。

腹主动脉 (`3, 33`)、下腔静脉 (`30`) 和门静脉 (`26, 27`) 同时具有器官和血管语义。它们出现在 `organ_labels` 时，仍按现有血管规则参与特征提取。

## 可能器官清单

`eus_possible_organs.json` 是裁剪结果根目录的单一清单文件，包含版本和排序后的 `organs` 数组。数组固定包含上表的 11 个规范器官，每项使用以下字段：

- `organ_label`：规范名称，例如 `portal_vein`。
- `eus_label_ids`：映射到该器官的所有原始 EUS ID，升序排列。
- `eus_label_names`：与 ID 一一对应的原始中文名称。
- `role`：`organ` 或 `organ_and_vessel`。
- `vessel_type`：仅双重角色时填写 `artery` 或 `vein`，其余为 `null`。
- `canonical_vessel_label_id`：仅门静脉填写 `26`，其余为 `null`；这表明 `27` 统一归并到 `26`。

因此 `aorta` 使用 ID `3, 33`、角色 `organ_and_vessel`、血管类型 `artery`；`inferior_vena_cava` 使用 ID `30`、角色 `organ_and_vessel`、血管类型 `vein`；`portal_vein` 使用 ID `26, 27`、角色 `organ_and_vessel`、血管类型 `vein`、规范血管 ID `26`。其余八个条目均为 `organ`。

## 数据流与归并规则

1. `process_cropped_folder()` 从裁剪后 TAR 读取 JSON 和可选 NIfTI。
2. 从 `FrameLabelModel.FrameLabel` 读取已声明的原始 ID；缺失、`null` 或空数组视为没有 JSON 声明，不回退到多边形。
3. 若 NIfTI 存在，从已转换为图像坐标的裁剪后标签图读取非零 ID；只保留固定器官映射中的 ID。
4. 两路 ID 各自映射为规范器官名称，再做稳定去重和字典序排序。
5. 详情 JSON 写入：
   - `organ_labels`
   - `organ_label_source: "frame_label_and_cropped_nifti"`
   - `frame_label_organ_ids`
   - `cropped_nifti_organ_ids`
6. 单帧图库 JSONL 同样写入 `organ_labels` 和 `organ_label_source`，但保留现有多器官不适合表达的 `organ: "unknown"` 字段以保持兼容。

JSON-only TAR 没有裁剪后 NIfTI 时，`cropped_nifti_organ_ids` 为空，器官集合仅来自 `FrameLabel`。

### 门静脉血管特征

血管特征提取仍输出下游兼容的 `label: "vein"` 或 `label: "artery"`。其中 NIfTI 标签 `26` 与 `27` 在连通域分析之前合成为一个门静脉二值掩膜：

- 结果的规范 `label_id` 为 `26`。
- 连通域可跨越原始 `26` 与 `27` 的相邻像素。
- 详情中的每个门静脉特征和跳过项新增 `source_label_ids`，记录该连通域使用的原始 ID 子集。
- 图库适配记录保持原有四字段特征格式，不暴露新增审计字段。

其他静脉和动脉保持现有标签分组、连通域、边缘剔除和颜色行为。`27` 在器官提取和血管特征中都不再作为独立的门静脉汇合部类别。

## 错误处理与兼容性

- NIfTI 维度或尺寸异常仍立即报错；批处理沿用暂存目录清理与原子发布。
- 缺失或格式异常的 `FrameLabelModel` 不报错，按无 JSON 声明处理。
- `ColorLabelTableModel` 只用于血管特征描述，不作为器官出现的来源。
- 既有 11 个每帧产物不增不减；只更新两个检索元数据文件的内容。
- 根目录额外生成一个 `eus_possible_organs.json`；它不影响每帧产物计数，也不依赖本批次出现的实际标签。

## 验证

新增或更新测试覆盖：

1. `FrameLabel` 中已声明、但 NIfTI 与多边形均未出现的器官仍进入 `organ_labels`。
2. 裁剪后 NIfTI 中出现、但 `FrameLabel` 未声明的器官仍进入 `organ_labels`。
3. 两路并集去重，胆囊、胆管、`S0`、非例外血管和未知 ID 被排除。
4. `3/33`、`30`、`26/27` 同时以器官和血管语义处理。
5. `26/27` 在血管特征中合成规范门静脉连通域，保留来源 ID 审计，触边剔除仍正确。
6. JSON-only TAR 仅使用 `FrameLabel`；端到端批处理保留 11 项产物并写入器官元数据。
7. 批处理根目录生成包含 11 个规范器官、双重血管角色和门静脉 `27 -> 26` 规范化信息的 `eus_possible_organs.json`。
8. 完整测试通过后，重新运行 `C:\\Users\\zhangyutang\\Desktop\\学姐标注EUS` 的固定 10 cm 裁剪，并验证桌面结果目录中的 105 帧、元数据一致性、器官清单和图像尺寸。
