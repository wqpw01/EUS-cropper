# EUS-cropper

固定 10 cm EUS 标注裁剪与可视化生成工具。

## 固定 10 cm 裁剪检索特征

`scripts/crop_picked_10cm.py` 会为每个成功裁剪的帧自动生成：

- `<frame>_cropped_retrieval_features.json`：完整血管连通域、被边界截断的连通域和坐标信息。
- `<frame>_cropped_gallery.jsonl`：可由兼容 `2021.py` 的二维血管检索流程加载的单帧记录。

检索特征只包含未接触裁剪边界的 8 连通域。静脉标签为 ID `26-32`，动脉标签为 ID `3` 和 `33-40`。坐标以裁剪图左上角为原点，x 向右、y 向下，裁剪平面范围为 100 mm x 100 mm；这些是合成二维坐标，不代表患者三维世界位姿。

每个帧目录包含原有九项裁剪产物以及上述两项检索文件。图库记录会引用 `*_cropped_vessel_label_white.png` 和 `*_cropped_vessel_overlay.png`，因此仅展示参与检索的血管边界。

### 器官元数据

每帧检索元数据的 `organ_labels` 是 JSON `FrameLabelModel.FrameLabel` 与裁剪后 NIfTI 实际标签的并集。胆囊、胆管和普通血管不进入器官集合；腹主动脉、下腔静脉和门静脉同时保留器官与血管语义。

门静脉汇合部 ID `27` 在器官标签和血管特征中均归并为门静脉 ID `26`。批处理根目录会生成 `eus_possible_organs.json`，固定列出全部 11 个可能出现的器官、其 EUS ID、中文名称以及血管双重角色。
