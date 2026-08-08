# 裁剪计划：1020真值 配对图片

## 调研结果

### 数据情况
- **输入路径**: `C:\Users\zhangyutang\Desktop\CT-EUS定位项目\数据\1020真值\配对图片`
- **输出路径**: `C:\Users\zhangyutang\Desktop\CT-EUS定位项目\数据\1020真值\裁剪后`
- **图像数量**: 22 张 PNG 图像
- **图像分辨率**: 全部 1920×1080
- **标注文件**: **无**（配对图片目录中没有 .tar 标注文件）
- **目录结构**: 扁平目录（无 group_XXX 子文件夹）

### 现有代码分析
- `batch_process.py` 期望 `group_XXX` 文件夹 + `.tar` 标注文件，与当前数据不匹配
- `cropper.py` 的核心裁剪功能可复用
- 裁剪坐标: 1920×1080 → `(576, 118, 1580, 985)` → 输出 1004×867

### 核心差异
| 项目 | 现有代码期望 | 实际数据 |
|------|-------------|---------|
| 目录结构 | group_XXX 子目录 | 扁平目录 |
| 标注文件 | .tar 伴随文件 | 无 |
| 处理内容 | 裁剪 + 标注转换 + 可视化 | 仅裁剪图像 |

## 实施方案

### 方案：编写简化裁剪脚本

由于数据只有图像没有标注，无需使用现有的 `batch_process.py`。编写一个简化的裁剪脚本，直接复用 `src/cropper.py` 的核心功能。

**具体步骤**:

1. **创建 `scripts/crop_simple.py`** - 一个简化的批量裁剪脚本:
   - 读取 `配对图片` 目录下所有 PNG 文件
   - 使用 `cropper.crop_image()` 进行裁剪
   - 保存到 `裁剪后` 目录，保持原始文件名
   - 输出处理统计信息

2. **修改 `src/config.py`** - 更新路径配置:
   - `INPUT_BASE_PATH` → 配对图片路径 (WSL 挂载格式)
   - `CROPPED_OUTPUT_PATH` → 裁剪后路径 (WSL 挂载格式)

3. **运行脚本** - 执行裁剪并验证结果

### 路径映射
- Windows: `C:\Users\zhangyutang\Desktop\CT-EUS定位项目\数据\1020真值\配对图片`
- WSL: `/mnt/c/Users/zhangyutang/Desktop/CT-EUS定位项目/数据/1020真值/配对图片`
- Windows: `C:\Users\zhangyutang\Desktop\CT-EUS定位项目\数据\1020真值\裁剪后`
- WSL: `/mnt/c/Users/zhangyutang/Desktop/CT-EUS定位项目/数据/1020真值/裁剪后`

### 预期输出
- 22 张裁剪后的图像 (1004×867)
- 保存至 `裁剪后` 目录，文件名与原始一致

### 风险评估
- 风险低：数据量小（22张），仅涉及图像裁剪操作
- 现有 `cropper.py` 模块经过测试，核心功能可靠
