#!/usr/bin/env python3
"""
批量处理脚本

对数据目录中有标注的图像进行裁剪处理：
1. 筛选掉没有有效多边形标注的图像
2. 保存裁剪后的图像
3. 保存转换后的标注 JSON
4. 生成可视化叠加图（裁剪后图像 + 标注多边形）
"""

import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm
from PIL import Image, ImageDraw

from src.config import INPUT_BASE_PATH, CROPPED_OUTPUT_PATH, calculate_crop_region
from src.utils import find_all_groups, find_image_label_pairs, extract_label_json
from src.cropper import crop_image
from src.label_processor import (
    transform_label_json,
    save_label_json,
    get_polygons_from_label,
    get_color_for_label
)


def has_valid_polygons(label_data: dict) -> bool:
    """
    检查标注数据中是否有有效的多边形

    Args:
        label_data: 标注 JSON 数据

    Returns:
        是否有有效多边形
    """
    if not label_data:
        return False

    polygons = get_polygons_from_label(label_data)
    return len(polygons) > 0


def draw_polygons_on_image(
    image: Image.Image,
    polygons: list[dict],
    label_data: dict,
    line_width: int = 2
) -> Image.Image:
    """
    在图像上绘制多边形标注

    Args:
        image: PIL 图像
        polygons: 多边形列表
        label_data: 原始标注数据（用于获取颜色）
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

        color = get_color_for_label(polygon["label"], label_data)

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


def process_single_item(args: tuple) -> dict:
    """
    处理单个图像-标注对（用于并行处理）

    只处理有有效多边形标注的图像

    Args:
        args: (img_path, label_path, output_dir)

    Returns:
        处理结果字典
    """
    img_path, label_path, output_dir = args

    result = {
        "image": img_path.name,
        "group": img_path.parent.name,
        "skipped": False,
        "skip_reason": None,
        "image_success": False,
        "label_success": False,
        "vis_success": False,
        "polygon_count": 0,
        "error": None
    }

    try:
        # 检查标注文件是否存在
        if not label_path or not label_path.exists():
            result["skipped"] = True
            result["skip_reason"] = "no_label_file"
            return result

        # 提取标注数据
        label_data = extract_label_json(label_path)
        if not label_data:
            result["skipped"] = True
            result["skip_reason"] = "label_parse_failed"
            return result

        # 检查是否有有效多边形
        if not has_valid_polygons(label_data):
            result["skipped"] = True
            result["skip_reason"] = "no_polygons"
            return result

        # 获取裁剪区域
        with Image.open(img_path) as img:
            width, height = img.size
        crop_region = calculate_crop_region(width, height)

        # 构建输出路径（保持相对路径结构）
        rel_path = img_path.relative_to(INPUT_BASE_PATH)
        output_img_path = output_dir / "images" / rel_path
        output_vis_path = output_dir / "visualizations" / rel_path

        # 裁剪图像
        success, _ = crop_image(img_path, output_img_path, crop_region)
        result["image_success"] = success

        if not success:
            result["error"] = "image_crop_failed"
            return result

        # 转换标注坐标
        transformed = transform_label_json(label_data, crop_region)
        polygons = get_polygons_from_label(transformed)
        result["polygon_count"] = len(polygons)

        # 保存标注 JSON
        label_rel_path = label_path.relative_to(INPUT_BASE_PATH)
        output_label_path = output_dir / "labels" / label_rel_path.with_suffix('.json')
        result["label_success"] = save_label_json(transformed, output_label_path)

        # 生成可视化图像
        cropped_img = Image.open(output_img_path)
        vis_img = draw_polygons_on_image(cropped_img, polygons, label_data)
        output_vis_path.parent.mkdir(parents=True, exist_ok=True)
        vis_img.save(output_vis_path)
        result["vis_success"] = True

    except Exception as e:
        result["error"] = str(e)

    return result


def run_batch_process(num_workers: int = None):
    """
    运行批量处理

    Args:
        num_workers: 并行进程数，默认使用 CPU 核心数
    """
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    print("=" * 60)
    print("EUS 图像批量裁剪处理")
    print("=" * 60)
    print(f"输入路径: {INPUT_BASE_PATH}")
    print(f"输出路径: {CROPPED_OUTPUT_PATH}")
    print(f"  - images/        裁剪后的图像")
    print(f"  - labels/        转换后的标注 JSON")
    print(f"  - visualizations/ 可视化叠加图")
    print(f"并行进程: {num_workers}")
    print("-" * 60)

    # 创建输出目录
    (CROPPED_OUTPUT_PATH / "images").mkdir(parents=True, exist_ok=True)
    (CROPPED_OUTPUT_PATH / "labels").mkdir(parents=True, exist_ok=True)
    (CROPPED_OUTPUT_PATH / "visualizations").mkdir(parents=True, exist_ok=True)

    # 收集所有待处理项
    groups = find_all_groups(INPUT_BASE_PATH)
    print(f"找到 {len(groups)} 个 group 文件夹")

    all_items = []
    for group in groups:
        pairs = find_image_label_pairs(group)
        for img_path, label_path in pairs:
            all_items.append((img_path, label_path, CROPPED_OUTPUT_PATH))

    total = len(all_items)
    print(f"共 {total} 个图像-标注对")
    print("-" * 60)
    print("开始处理（自动跳过无标注的图像）...")

    # 并行处理
    stats = {
        "total": total,
        "processed": 0,
        "skipped_no_label": 0,
        "skipped_no_polygons": 0,
        "skipped_other": 0,
        "image_success": 0,
        "label_success": 0,
        "vis_success": 0,
        "total_polygons": 0,
        "errors": []
    }

    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_item, all_items),
            total=total,
            desc="处理进度"
        ))

    # 统计结果
    for r in results:
        if r["skipped"]:
            if r["skip_reason"] == "no_label_file":
                stats["skipped_no_label"] += 1
            elif r["skip_reason"] == "no_polygons":
                stats["skipped_no_polygons"] += 1
            else:
                stats["skipped_other"] += 1
        else:
            stats["processed"] += 1
            if r["image_success"]:
                stats["image_success"] += 1
            if r["label_success"]:
                stats["label_success"] += 1
            if r["vis_success"]:
                stats["vis_success"] += 1
            stats["total_polygons"] += r["polygon_count"]

        if r["error"]:
            stats["errors"].append({
                "image": r["image"],
                "error": r["error"]
            })

    # 输出统计
    print("\n" + "=" * 60)
    print("处理完成统计:")
    print("-" * 60)
    print(f"  总图像数: {stats['total']}")
    print(f"  有效处理: {stats['processed']}")
    print(f"  跳过（无标注文件）: {stats['skipped_no_label']}")
    print(f"  跳过（无多边形标注）: {stats['skipped_no_polygons']}")
    print(f"  跳过（其他原因）: {stats['skipped_other']}")
    print("-" * 60)
    print(f"  图像裁剪成功: {stats['image_success']}")
    print(f"  标注转换成功: {stats['label_success']}")
    print(f"  可视化生成成功: {stats['vis_success']}")
    print(f"  总多边形数: {stats['total_polygons']}")

    if stats["errors"]:
        print(f"\n错误列表 ({len(stats['errors'])} 个):")
        for err in stats["errors"][:10]:
            print(f"  - {err['image']}: {err['error']}")
        if len(stats["errors"]) > 10:
            print(f"  ... 还有 {len(stats['errors']) - 10} 个错误")

    print("\n" + "=" * 60)
    print(f"输出目录: {CROPPED_OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EUS 图像批量裁剪（仅处理有标注的）")
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=None,
        help="并行进程数（默认 CPU 核心数 - 1）"
    )

    args = parser.parse_args()
    run_batch_process(args.workers)
