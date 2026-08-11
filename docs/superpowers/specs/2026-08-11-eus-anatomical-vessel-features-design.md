# EUS 三类解剖血管特征与可视化设计

## 目标

固定 10 cm EUS 裁剪流程继续输出原有的通用静脉和动脉检索特征，同时为每帧增加下列三类解剖血管的独立连通域特征及合并可视化：

1. 下腔静脉（IVC）。
2. 腹主动脉（Ao）。
3. 门静脉系（PV），其中包括门静脉、门静脉汇合部、肠系膜上静脉和脾静脉。

本功能只接入 `scripts/crop_picked_10cm.py` 的固定 10 cm 流程。既有裁剪区域、器官元数据、通用静脉/动脉特征、通用血管可视化和图库 JSONL 保持兼容。用户已取消远程服务器同步；完成后只推送本地 Git 仓库到 GitHub `origin`。

## 方案选择

采用在现有单帧检索详情 JSON 中增加独立字段的方案：

- 原有 `features` 和 `skipped_components` 继续只表示通用 `vein` / `artery` 特征，保持现有检索消费者的输入契约。
- 新增 `anatomical_vessel_features` 与 `anatomical_vessel_skipped_components`，只表示 IVC、Ao、PV 三类特征。
- 新增三张同时绘制三类血管的图，不为每种血管分别新增图片或 JSON 文件。

不将新类别混入原有 `features`，以避免下游仅理解通用静脉/动脉类别的程序发生行为变化；也不创建第二份 JSON，以避免同一帧的检索信息分散在多份文件中。

## 固定标签与颜色

| 解剖血管类 | 规范 `label` | 规范 `label_id` | 原始 EUS 标签 ID | 边界 RGB |
| --- | --- | ---: | --- | --- |
| 腹主动脉（Ao） | `aorta` | `3` | `3, 33` | `(255, 0, 0)` 红色 |
| 下腔静脉（IVC） | `inferior_vena_cava` | `30` | `30` | `(0, 0, 255)` 蓝色 |
| 门静脉系（PV） | `portal_venous_system` | `26` | `26, 27, 28, 29` | `(170, 85, 255)` 紫色 |

`28` 是肠系膜上静脉，`29` 是脾静脉。门静脉系掩膜在连通域分析前按四个来源 ID 的并集建立；相邻的不同来源标签可成为同一个门静脉系连通域。肝静脉 `31`、肾静脉 `32` 和其他动静脉不属于新三类，但仍参与原有通用静脉/动脉流程。

## 每帧新增产物

原有每帧 11 个文件保留。每帧再新增以下三个 PNG，因此每帧总计 14 个文件：

1. `<frame>_original_ivc_ao_pv_overlay.png`：原始 `1920 x 1080` 图像叠加三类血管边界。
2. `<frame>_cropped_ivc_ao_pv_overlay.png`：裁剪后的 `960 x 960` 图像叠加三类血管边界。
3. `<frame>_cropped_ivc_ao_pv_label_white.png`：纯白 `960 x 960` 背景叠加三类血管边界。

三张图均只绘制边界线，不填充标签区域。白底图只允许纯白和上表定义的红、蓝、紫三种边界色。原图叠加图使用原始 JSON 坐标；两张裁剪图使用已按固定裁剪区域变换的 JSON 坐标。原有通用静脉/动脉可视化继续使用其既有青蓝/橙红配色和文件名。

## 检索详情 JSON

`<frame>_cropped_retrieval_features.json` 的 `schema_version` 从 `cropped-retrieval-features/v1` 升为 `cropped-retrieval-features/v2`，并新增以下字段：

```json
{
  "anatomical_vessel_visualizations": {
    "original_overlay_png": "<frame>_original_ivc_ao_pv_overlay.png",
    "cropped_overlay_png": "<frame>_cropped_ivc_ao_pv_overlay.png",
    "boundary_only_png": "<frame>_cropped_ivc_ao_pv_label_white.png"
  },
  "anatomical_vessel_features": [],
  "anatomical_vessel_skipped_components": []
}
```

每个完整特征包含下列信息：

- `label`、`label_id`、固定中文 `label_desc`。
- `source_label_ids`：该连通域实际出现的原始 ID 子集，而非只写类别的全部候选 ID。
- `component_index`、`area_px`、`centroid_px`。
- `x_mm`、`y_mm`、`area_mm2`，坐标系仍为裁剪图左上角原点、x 向右、y 向下。

跳过项使用同一基础字段，并以 `reason: "touches_image_edge"` 说明为什么不参与检索。已有 `features`、`skipped_components`、器官字段和 `adapter_record` 保持其原有含义。`<frame>_cropped_gallery.jsonl` 不新增三类特征或新的 PNG 引用，继续作为只含通用静脉/动脉适配特征的兼容接口。

## 连通域与边界规则

用户确认“不闭合”仅指血管被 10 cm 裁剪边界截断。原始 EUS 血管标签存于 `Polys.Shapes`，没有可靠的 `closed` / `open` 字段；因此不能也不会根据 JSON 多边形闭合标记过滤特征。三个新类别均使用与既有流程相同的 8 连通域规则。任何一个连通域只要接触裁剪后标签图的上、下、左或右边缘，即：

1. 不写入 `anatomical_vessel_features`。
2. 写入 `anatomical_vessel_skipped_components`。
3. 仍在两张裁剪可视化中显示其边界；原图叠加图也始终显示原始轮廓。

因此，特征检索只使用完整保留的血管，而图像结果始终保留被裁剪截断的边界作为人工审计依据。JSON-only 标签 TAR 没有 NIfTI 时，新增特征和跳过列表为空，但三张 JSON 多边形可视化仍照常生成。

## 数据流与错误处理

1. 批处理读取原始 JPG 和标签 TAR，生成已有裁剪图、裁剪 TAR 和全部已有产物。
2. 同时从原始和变换后的标签 JSON 中筛出上述来源 ID，按类别及固定颜色绘制三张新增 PNG。
3. `process_cropped_folder()` 从裁剪后 TAR 的 NIfTI 标签图提取通用特征和新增三类特征。
4. 检索详情写入新增字段；图库 JSONL 继续使用原有通用特征和通用血管图引用。
5. 固定批处理在生成后校验两张新增裁剪 PNG 均存在且尺寸与裁剪标签图一致；`process_cropped_folder()` 仍与现有行为一致，仅在独立调用时已存在这些图才校验其尺寸。尺寸不匹配或标签 TAR 异常时立即报错。

批处理继续使用暂存目录和原子替换。输出目录已存在、暂存目录已存在、输入图像尺寸不符或任一帧失败时，不发布半成品结果；失败时清理本次暂存目录。

## 测试与验收

测试覆盖以下行为：

1. 通用静脉/动脉 `features` 和 `skipped_components` 的既有结果不变。
2. IVC `30`、Ao `3/33`、PV `26/27/28/29` 分别生成正确的规范类别、规范 ID 和实际 `source_label_ids`。
3. 同一类别中相邻的来源标签按 8 连通规则合并；不相邻区域保留为独立连通域。
4. 四个方向的触边连通域仅进入新增跳过列表。
5. 端到端批处理生成 14 个帧内文件、三张新增 PNG 的正确尺寸，以及白底图严格使用白、红、蓝、紫四种颜色。
6. 完整 `pytest -q` 通过，且 Git 工作树只包含本功能相关变更。
7. 重新运行 `C:\\Users\\zhangyutang\\Desktop\\学姐标注EUS` 的固定 10 cm 裁剪；确认输出根目录、帧数、每帧文件数、代表性可视化和 JSON 特征映射正确。
8. 验证通过后提交实现、测试和文档，并推送到 GitHub `origin/main`。
