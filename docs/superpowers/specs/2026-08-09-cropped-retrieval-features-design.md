# 固定 10 cm 裁剪检索特征设计

## 目标

将二维血管检索特征提取固定接入 `scripts/crop_picked_10cm.py`。每个成功裁剪的帧目录在保留已有九项裁剪产物的基础上，自动新增可供 `2021.py` 兼容检索加载的特征详情和图库记录。

本功能只适用于固定 10 cm 裁剪流程；不修改 `crop_simple.py`、`batch_process.py`，也不回写已有的桌面裁剪结果。

## 方案选择

采用项目内模块加直接调用的方式：新增 `src/cropped_retrieval.py`，由 `scripts/crop_picked_10cm.py` 在裁剪标签 TAR 写入完成后调用。模块不依赖 `/home/zyt/ct_vascular_resampling` 的代码或路径，复现其 `extract_cropped_retrieval_features.py` 的数据语义。

不采用跨项目调用脚本，避免部署时依赖外部工作树和环境；不把特征逻辑内联到批处理脚本，避免难以独立测试和复用。

## 数据流

1. 固定 10 cm 流程生成裁剪图、可视化产物和 `<frame>_cropped_jpg_Label.tar`。
2. `process_cropped_folder(frame_dir, width_mm=100.0, length_mm=100.0)` 从该 TAR 读取唯一 JSON 与可选的单个 NIfTI 标签图。
3. NIfTI 数据转换为图像坐标的 `[y, x]` 数组，保持与参考脚本的 SimpleITK 读取方向一致；只使用本项目已有的 `nibabel` 读取 NIfTI。
4. 对动静脉标签按 8 连通域提取完整截面。任一像素触及图像四边的连通域不写入检索特征，并在详情文件中说明 `touches_image_edge`。
5. 在同一帧目录写入两份检索文件。批处理仅在所有帧成功后，将暂存目录原子替换为目标目录；特征提取错误沿用现有回滚行为。

## 标签与坐标规则

- 静脉标签 ID：`26` 至 `32`。
- 动脉标签 ID：`3`、`33` 至 `40`。ID `3` 在当前 EUS 数据中为“腹主动脉”，用户已确认纳入。
- 使用 8 连通域，按标签 ID 升序处理，静脉组先于动脉组。
- 960 x 960 像素覆盖 100 mm x 100 mm。像素间距按参考脚本使用端点映射：`100.0 / 959.0 mm/px`。
- 每个保留连通域记录标签类别、ID、标签描述、连通域序号、像素面积、像素质心、毫米质心和平方毫米面积。毫米坐标系为左上角原点、x 向右、y 向下。

## 输出契约

每个帧目录新增：

- `<frame>_cropped_retrieval_features.json`
  - 包含 schema 版本、输入 TAR、标签来源、图像尺寸、裁剪物理尺寸、像素间距、完整特征、跳过的连通域和图库记录。
- `<frame>_cropped_gallery.jsonl`
  - 一行兼容 `2021.py` 的记录。存在完整血管特征时 `status` 为 `gallery`，否则为 `unindexed`。

图库记录使用 `synthetic_2d_10cm_crop` 坐标系，明确 `patient_world_pose: false`，不把二维裁剪结果误作患者三维位姿。它引用当前流程已有的：

- `ct_png`：`<frame>_cropped.jpg`
- `boundary_only_png`：`<frame>_cropped_vessel_label_white.png`
- `ct_overlay_png`：`<frame>_cropped_vessel_overlay.png`

没有 NIfTI 的 JSON-only 标签 TAR 被视为零掩膜：仍生成两份检索文件和白底标签图，不生成检索特征，状态为 `unindexed`。

批处理返回值在原有 `total` 和 `processed` 之外新增 `gallery_records`，用于汇报含至少一个可检索完整血管截面的帧数。每帧产物总数由九个变为十一个。

## 依赖与错误处理

新增 `scipy` 以使用 `scipy.ndimage.label` 完成 8 连通域分析；NIfTI 继续使用现有 `nibabel` 依赖，避免新增 SimpleITK。

标签 TAR 必须包含一个 JSON 和至多一个 `.nii` 或 `.nii.gz`。JSON 缺失、多个 NIfTI、NIfTI 维度不符、标签尺寸不符或既有可视化尺寸不符均立即报错，并触发批处理暂存目录清理。不会写根目录汇总 JSONL，也不会覆盖用户已存在的目标输出目录。

## 验证

测试覆盖：

1. NIfTI 坐标转置与毫米坐标换算。
2. 静脉、动脉 ID `3`、动脉 ID `33` 的完整连通域提取。
3. 触边连通域被记录但不进入检索特征，非血管标签被忽略。
4. JSON-only TAR 生成 `unindexed` 空记录。
5. 固定 10 cm 批处理端到端生成十一项产物并汇报 `gallery_records`。
6. 完整测试套件通过，且无根目录检索汇总文件。
