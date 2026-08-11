# EUS-cropper

固定 10 cm EUS 标注裁剪与可视化生成工具。

## 固定 10 cm 裁剪检索特征

`scripts/crop_picked_10cm.py` 会为每个成功裁剪的帧自动生成：

- `<frame>_cropped_retrieval_features.json`：通用静脉/动脉检索特征，以及三类解剖血管的完整连通域、被边界截断的连通域和坐标信息。
- `<frame>_cropped_gallery.jsonl`：可由兼容 `2021.py` 的二维血管检索流程加载的单帧记录。

检索特征只包含未接触裁剪边界的 8 连通域。静脉标签为 ID `26-32`，动脉标签为 ID `3` 和 `33-40`。坐标以裁剪图左上角为原点，x 向右、y 向下，裁剪平面范围为 100 mm x 100 mm；这些是合成二维坐标，不代表患者三维世界位姿。

每个帧目录包含 14 个文件：原有九项裁剪产物、两项检索文件，以及下列三张合并解剖血管边界图：

- `<frame>_original_ivc_ao_pv_overlay.png`：原图叠加 Ao、IVC 和门静脉系边界。
- `<frame>_cropped_ivc_ao_pv_overlay.png`：裁剪图叠加三类边界。
- `<frame>_cropped_ivc_ao_pv_label_white.png`：纯白背景叠加三类边界。

Ao 使用红色 `RGB (255, 0, 0)`，IVC 使用蓝色 `RGB (0, 0, 255)`，门静脉系使用紫色 `RGB (170, 85, 255)`。门静脉系固定合并 ID `26, 27, 28, 29`，其中 `28` 为肠系膜上静脉、`29` 为脾静脉。三张新图始终绘制全部对应边界；只有触及裁剪边缘、即被截断的连通域会从 `anatomical_vessel_features` 中排除，并记录在 `anatomical_vessel_skipped_components`。

检索详情 JSON 的 schema 为 `cropped-retrieval-features/v2`。原有通用 `features`、`skipped_components` 和图库 JSONL 接口保持只表示静脉/动脉；新增 `anatomical_vessel_features`、`anatomical_vessel_skipped_components` 和 `anatomical_vessel_visualizations` 只表示 Ao、IVC、门静脉系。图库记录仍引用 `*_cropped_vessel_label_white.png` 和 `*_cropped_vessel_overlay.png`，以保持下游兼容。

### 器官元数据

每帧检索元数据的 `organ_labels` 是 JSON `FrameLabelModel.FrameLabel` 与裁剪后 NIfTI 实际标签的并集。胆囊、胆管和普通血管不进入器官集合；腹主动脉、下腔静脉和门静脉同时保留器官与血管语义。

门静脉汇合部 ID `27` 在器官标签和血管特征中均归并为门静脉 ID `26`。批处理根目录会生成 `eus_possible_organs.json`，固定列出全部 11 个可能出现的器官、其 EUS ID、中文名称以及血管双重角色。
