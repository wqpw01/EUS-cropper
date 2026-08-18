# 固定 10 cm EUS 裁剪复现包整理设计

## 目标

将 EUS-cropper 的当前 `main` 整理为供审稿人和合作方复现的最小代码包。仓库只提供固定 10 cm EUS 图像与配套标签的完整裁剪流程，不包含患者数据、旧版裁剪流程或开发过程文档。

Git 历史保留；当前工作树和远程 `main` 只呈现复现所需内容。

## 范围

保留唯一入口 `scripts/crop_picked_10cm.py`，以及其依赖的以下模块：

- `src/config.py`：裁剪区域值对象。
- `src/picked_10cm.py`：固定坐标图像与 NIfTI 裁剪、标签 TAR 重打包。
- `src/label_processor.py`：JSON 多边形变换与边界裁剪。
- `src/utils.py`：输入图像和标签 TAR 配对。
- `src/cropped_retrieval.py`：器官元数据、通用血管和三类解剖血管检索特征。

保留上述功能对应的测试，删除旧版通用裁剪、无标签裁剪和仅用于人工调试可视化的脚本与测试。删除早期开发计划和设计文档。

## 命令行接口

主脚本要求显式提供：

```bash
python scripts/crop_picked_10cm.py --input-dir INPUT_DIR --output-dir OUTPUT_DIR
```

输入目录为扁平目录，每个 `1920x1080` 的 PNG/JPEG 图像必须有同目录标签 TAR。输出目录必须不存在；流程在临时目录完成后原子替换为最终目录，避免半成品结果。

## 复现材料

仓库保留：

- `README.md`：英文主说明，面向审稿人和合作方。
- `README_zh-CN.md`：与英文说明等价的中文说明。
- `environment.yml`：使用 mamba 创建的固定 Python 环境。
- `requirements.txt`：与环境文件一致的 Python 依赖清单。
- `.gitignore`：排除原始/裁剪数据、环境、缓存和本地工具状态。

README 说明环境创建、输入规范、运行命令、输出文件、标签与颜色、特征完整性规则、验证命令、数据可用性和限制条件。医疗图像和标注不进入 Git。

## 验证

测试覆盖固定裁剪范围、JSON 与 NIfTI 标签变换、输入配对、批处理结果、通用血管特征、三类解剖血管特征以及器官目录。整理后测试必须在 `environment.yml` 所定义的 mamba 环境中通过。

## 非目标

- 不重写 Git 历史，不强制推送。
- 不提供原生 Windows 运行保证；文档将覆盖 Windows 的 WSL Ubuntu 和 Linux/macOS shell 用法。
- 不添加或分发任何真实医学影像、标注或患者信息。
- 不改变固定 10 cm 裁剪、标签语义、输出格式或检索特征定义。
