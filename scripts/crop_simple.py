#!/usr/bin/env python3
"""
简化批量裁剪脚本

对扁平目录中的所有 PNG 图像进行裁剪（无需标注文件）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm
from PIL import Image

from src.config import INPUT_BASE_PATH, CROPPED_OUTPUT_PATH, calculate_crop_region
from src.cropper import crop_image


def run_simple_crop():
    print("=" * 60)
    print("EUS 图像简化批量裁剪")
    print("=" * 60)
    print(f"输入路径: {INPUT_BASE_PATH}")
    print(f"输出路径: {CROPPED_OUTPUT_PATH}")
    print("-" * 60)

    # 收集所有 PNG 文件
    image_files = sorted(INPUT_BASE_PATH.glob("*.png"))
    total = len(image_files)
    print(f"找到 {total} 张图像")

    if total == 0:
        print("未找到图像文件，退出")
        return

    # 确认裁剪区域
    with Image.open(image_files[0]) as img:
        width, height = img.size
    crop_region = calculate_crop_region(width, height)
    print(f"裁剪区域: {crop_region.as_tuple()} → 输出 {crop_region.width}×{crop_region.height}")
    print("-" * 60)

    # 创建输出目录
    CROPPED_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    # 处理
    success_count = 0
    error_count = 0

    for img_path in tqdm(image_files, desc="裁剪进度"):
        output_path = CROPPED_OUTPUT_PATH / img_path.name
        success, _ = crop_image(img_path, output_path, crop_region)
        if success:
            success_count += 1
        else:
            error_count += 1
            print(f"\n  裁剪失败: {img_path.name}")

    # 统计
    print("\n" + "=" * 60)
    print("处理完成:")
    print(f"  成功: {success_count}")
    print(f"  失败: {error_count}")
    print(f"  输出目录: {CROPPED_OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    run_simple_crop()
