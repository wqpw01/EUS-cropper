# EUS-cropper

English: [README.md](README.md)

## 用途

EUS-cropper 是更大研究流程中的固定 10 cm EUS 图像与标签裁剪环节。它从一个含标注 EUS 帧的目录中生成配对图像、标注、可视化、器官元数据和二维血管检索结果。

## 本仓库功能

唯一公开工作流为：

```bash
python scripts/crop_picked_10cm.py --input-dir /path/to/input --output-dir /path/to/output
```

该流程将每一帧裁剪到固定像素范围 `(603, 123, 1563, 1083)`，转换配套标签 TAR，必要时裁剪其中的二维 NIfTI 标签图像，生成可视化结果，并提取血管检索元数据。

## 仓库范围与数据可用性

本仓库是面向审稿人和合作方的复现包，只包含源代码、测试和环境定义，不包含医学图像、标注、患者信息或已生成的裁剪结果。请将私有输入数据放在仓库外，或放在 Git 忽略的目录中。

## 环境要求

- Linux、macOS，或具备 Ubuntu 兼容 shell 的 Windows Subsystem for Linux（WSL）。
- [Mamba](https://mamba.readthedocs.io/) 或兼容的 conda 安装。
- 满足下述约定的输入目录。

尚未验证原生 Windows Python。在 Windows 上请使用 WSL 路径，例如 `/mnt/c/Users/<user>/...`，而不要使用反斜杠分隔的 Windows 路径。

## 安装

克隆或解压仓库后，在仓库根目录运行：

```bash
mamba env create --file environment.yml
mamba activate eus-cropper
```

提交的环境使用 Python 3.12.10，以及固定版本的 Pillow、NumPy、NiBabel、SciPy、tqdm 和 pytest。`requirements.txt` 提供对应的仅运行时 pip 依赖。

## 输入数据约定

`--input-dir` 必须是扁平目录，不会扫描子目录。每张受支持图像必须同时满足以下条件：

- 图像扩展名为 `.png`、`.jpg` 或 `.jpeg`。
- 图像尺寸严格为 `1920 x 1080` 像素。
- 同一目录中存在配对 TAR 文件。
- 对于 `frame_00000001.jpg`，预期 TAR 名为 `frame_00000001_jpg_Label.tar`。
- 若图像 stem 自身含有句点，实现还接受将 stem 中句点替换为下划线后的同名规则。
- TAR 内含一个标签 JSON；还可以含有一个二维 `.nii` 或 `.nii.gz` 标签图像。

JSON 保存用于叠加图的多边形边界。若存在 NIfTI 成员，流程会同时裁剪它并用于连通域检索特征。开始创建输出前会预检全部图像/TAR 对；缺少配对文件、图像尺寸不支持或 JSON 不可读都会中止整个批次。

## 运行完整固定 10 cm 裁剪

选择一个尚不存在的输出路径：

```bash
python scripts/crop_picked_10cm.py \
  --input-dir /path/to/input \
  --output-dir /path/to/eus_10cm_results
```

流程会创建名为 `.<output-name>.in_progress` 的同级临时目录；只有所有帧都成功后才会替换为目标输出目录。若输出目录或临时目录已存在，程序会报错而不会覆盖其中内容。

裁剪画布为 `960 x 960` 像素。源图 `y=123` 至 `y=1079` 填充前 957 行，最后 3 行为白色图像填充和零值 NIfTI 填充。JSON 多边形坐标也会平移并裁剪到相同的固定区域。

## 输出结构

输出根目录中每个输入帧各有一个文件夹，另有根级 `eus_possible_organs.json`。每个帧文件夹包含 14 个文件：

| 文件模式 | 含义 |
| --- | --- |
| `<frame>.<ext>` | 原图的逐字节副本。 |
| `<frame>_<ext>_Label.tar` | 原标签 TAR 的逐字节副本。 |
| `<frame>_cropped.<ext>` | 固定 10 cm、`960 x 960` 的裁剪图。 |
| `<frame>_cropped_<ext>_Label.tar` | 裁剪后的 JSON 标签 TAR；若原始 TAR 中有 NIfTI，也包含裁剪 NIfTI。 |
| `<frame>_original_overlay.png` | 原图叠加全部原始标签边界。 |
| `<frame>_cropped_overlay.png` | 裁剪图叠加全部裁剪后标签边界。 |
| `<frame>_cropped_label_white.png` | 白色背景叠加全部裁剪后标签边界。 |
| `<frame>_cropped_vessel_overlay.png` | 裁剪图叠加通用动脉/静脉边界。 |
| `<frame>_cropped_vessel_label_white.png` | 白色背景叠加通用动脉/静脉边界。 |
| `<frame>_original_ivc_ao_pv_overlay.png` | 原图叠加腹主动脉、下腔静脉和门静脉系边界。 |
| `<frame>_cropped_ivc_ao_pv_overlay.png` | 裁剪图叠加腹主动脉、下腔静脉和门静脉系边界。 |
| `<frame>_cropped_ivc_ao_pv_label_white.png` | 白色背景叠加腹主动脉、下腔静脉和门静脉系边界。 |
| `<frame>_cropped_retrieval_features.json` | 器官标签、通用血管特征、三类解剖血管特征与跳过连通域。 |
| `<frame>_cropped_gallery.jsonl` | 下游二维血管图库所用的一条兼容记录。 |

## 标签语义与血管颜色

通用血管可视化将 EUS 标签 `26-32` 归为静脉，将 `3, 33-40` 归为动脉：

- 静脉：青蓝色，`RGB (0, 188, 212)`。
- 动脉：橙红色，`RGB (255, 82, 0)`。

三类解剖血管可视化使用另一套固定颜色：

| 解剖分组 | EUS 标签 ID | 边界颜色 |
| --- | --- | --- |
| 腹主动脉（Ao） | `3, 33` | 红色，`RGB (255, 0, 0)` |
| 下腔静脉（IVC） | `30` | 蓝色，`RGB (0, 0, 255)` |
| 门静脉系（PV） | `26, 27, 28, 29` | 紫色，`RGB (170, 85, 255)` |

门静脉系包括门静脉及分支、门静脉汇合部、肠系膜上静脉和脾静脉。`eus_possible_organs.json` 列出批处理结果中可能出现的 EUS 器官标签；腹主动脉、下腔静脉和门静脉同时保留器官与血管角色。

## 检索特征与裁剪边界规则

`cropped-retrieval-features/v2` 同时记录通用动脉/静脉和 Ao/IVC/PV 三类解剖血管特征。对于 NIfTI 标签，流程采用八连通二维连通域。任何接触 `960 x 960` 裁剪图像边缘的连通域都会被视为被裁剪截断：

- 不进入 `features` 或 `anatomical_vessel_features`。
- 以 `reason: "touches_image_edge"` 保留在对应的跳过连通域列表中。
- 对应 JSON 边界仍会画在所有适用的边界可视化图中。

该规则只影响特征是否纳入，不会从全标签、通用血管或 Ao/IVC/PV 叠加图中删除可见多边形边界。若输入 TAR 不含 NIfTI，图像/JSON 裁剪和可视化仍会生成，但基于 NIfTI 的特征数组不含连通域。

所有特征位置都是合成的二维裁剪平面坐标：原点为裁剪图左上角，`x` 向右增加，`y` 向下增加，名义范围为 `100 mm x 100 mm`。这些不是患者三维空间坐标。

## 验证

激活环境后运行：

```bash
pytest -q
```

测试套件使用合成的图像、TAR、JSON 和 NIfTI fixture，验证固定裁剪范围、白色填充、JSON 裁剪、NIfTI 几何保持、输入配对、输出产物、器官元数据、通用血管检索特征以及 Ao/IVC/PV 特征行为。

## 限制

- 公开工作流仅接受扁平的 `1920 x 1080` 输入目录。
- 裁剪坐标固定，运行时不可配置。
- 代码只处理二维图像和 NIfTI 标签。
- 本包不分发原始医学数据。
- 原生 Windows Python 不属于已验证的平台范围。

## 引用

使用本复现包时请引用相关论文。此仓库只是该论文流程中的一个环节，因此未在此定义仓库 DOI 或作者列表。
