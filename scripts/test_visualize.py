#!/usr/bin/env python3
"""
测试可视化脚本

对前 N 张图像进行裁剪，并在裁剪后的图像上绘制标注多边形，
生成可视化结果供人工验证。
"""

import sys
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image, ImageDraw

from src.config import INPUT_BASE_PATH, TEST_VIS_PATH
from src.utils import find_all_groups, find_image_label_pairs, extract_label_json
from src.cropper import crop_image_in_memory
from src.label_processor import (
    transform_label_json,
    get_polygons_from_label,
    get_color_for_label
)


def draw_polygons_on_image(
    image: Image.Image,
    polygons: list[dict],
    line_width: int = 2
) -> Image.Image:
    """
    在图像上绘制多边形标注

    Args:
        image: PIL 图像
        polygons: 多边形列表
        line_width: 线宽

    Returns:
        绘制了多边形的图像
    """
    # 转换为 RGB 模式以支持彩色绘制
    if image.mode != 'RGB':
        image = image.convert('RGB')

    draw = ImageDraw.Draw(image)

    for polygon in polygons:
        points = polygon["points"]
        if len(points) < 2:
            continue

        # 转换为 PIL 需要的格式 [(x1, y1), (x2, y2), ...]
        xy_points = [(p[0], p[1]) for p in points]

        color = get_color_for_label(polygon["label"], {})

        # 绘制多边形轮廓
        if polygon.get("closed", False):
            draw.polygon(xy_points, outline=color, width=line_width)
        else:
            draw.line(xy_points, fill=color, width=line_width)

        # 绘制顶点
        for pt in xy_points:
            draw.ellipse(
                [pt[0] - 3, pt[1] - 3, pt[0] + 3, pt[1] + 3],
                fill=color
            )

    return image


def run_test(num_images: int = 5):
    """
    运行测试可视化

    Args:
        num_images: 测试图像数量
    """
    print(f"开始测试可视化，将处理前 {num_images} 张图像")
    print(f"输入路径: {INPUT_BASE_PATH}")
    print(f"输出路径: {TEST_VIS_PATH}")
    print("-" * 50)

    # 创建输出目录
    TEST_VIS_PATH.mkdir(parents=True, exist_ok=True)

    # 获取所有 group
    groups = find_all_groups(INPUT_BASE_PATH)
    print(f"找到 {len(groups)} 个 group 文件夹")

    processed = 0
    success = 0

    for group in groups:
        if processed >= num_images:
            break

        print(f"\n处理 {group.name}...")
        pairs = find_image_label_pairs(group)

        for img_path, label_path in pairs:
            if processed >= num_images:
                break

            print(f"  [{processed + 1}/{num_images}] {img_path.name}")

            # 裁剪图像
            cropped_img, crop_region = crop_image_in_memory(img_path)
            if cropped_img is None:
                print(f"    图像裁剪失败")
                processed += 1
                continue

            print(f"    裁剪区域: {crop_region.as_tuple()}")
            print(f"    输出尺寸: {crop_region.width} x {crop_region.height}")

            # 处理标注
            if label_path:
                label_data = extract_label_json(label_path)
                if label_data:
                    # 转换坐标
                    transformed_label = transform_label_json(label_data, crop_region)

                    # 提取多边形
                    polygons = get_polygons_from_label(transformed_label)
                    print(f"    多边形数量: {len(polygons)}")

                    # 在裁剪后的图像上绘制多边形
                    vis_img = draw_polygons_on_image(cropped_img, polygons)

                    # 保存可视化结果
                    output_name = f"{group.name}_{img_path.stem}_vis.png"
                    output_path = TEST_VIS_PATH / output_name
                    vis_img.save(output_path)
                    print(f"    已保存: {output_path}")

                    success += 1
                else:
                    print(f"    标注解析失败")
                    # 仍然保存裁剪后的图像（无标注）
                    output_name = f"{group.name}_{img_path.stem}_vis.png"
                    cropped_img.save(TEST_VIS_PATH / output_name)
            else:
                print(f"    无对应标注文件")
                # 保存裁剪后的图像（无标注）
                output_name = f"{group.name}_{img_path.stem}_vis.png"
                cropped_img.save(TEST_VIS_PATH / output_name)

            processed += 1

    print("\n" + "=" * 50)
    print(f"测试完成: 处理 {processed} 张，成功 {success} 张")
    print(f"可视化结果保存在: {TEST_VIS_PATH}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EUS 裁剪测试可视化")
    parser.add_argument(
        "-n", "--num",
        type=int,
        default=5,
        help="测试图像数量（默认 5）"
    )

    args = parser.parse_args()
    run_test(args.num)
