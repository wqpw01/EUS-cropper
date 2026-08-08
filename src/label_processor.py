"""
标注处理模块

处理标注 JSON 中的多边形坐标，进行裁剪偏移
"""

import json
import copy
from pathlib import Path

from .config import CropRegion


def _is_inside_rect(point: list, width: int, height: int) -> bool:
    return 0 <= point[0] <= width and 0 <= point[1] <= height


def _is_inside_edge(point: list, edge: str, width: int, height: int) -> bool:
    if edge == "left":
        return point[0] >= 0
    if edge == "right":
        return point[0] <= width
    if edge == "top":
        return point[1] >= 0
    return point[1] <= height


def _intersect_edge(start: list, end: list, edge: str, width: int, height: int) -> list:
    dx = end[0] - start[0]
    dy = end[1] - start[1]

    if edge == "left":
        t = (0 - start[0]) / dx
        x = 0.0
        y = start[1] + t * dy
    elif edge == "right":
        t = (width - start[0]) / dx
        x = float(width)
        y = start[1] + t * dy
    elif edge == "top":
        t = (0 - start[1]) / dy
        x = start[0] + t * dx
        y = 0.0
    else:
        t = (height - start[1]) / dy
        x = start[0] + t * dx
        y = float(height)

    z_start = start[2] if len(start) > 2 else 0.0
    z_end = end[2] if len(end) > 2 else 0.0
    z = z_start + t * (z_end - z_start)
    return [x, y, z]


def _clip_against_edge(points: list, edge: str, width: int, height: int) -> list:
    if not points:
        return []

    clipped = []
    previous = points[-1]
    previous_inside = _is_inside_edge(previous, edge, width, height)

    for current in points:
        current_inside = _is_inside_edge(current, edge, width, height)

        if current_inside:
            if not previous_inside:
                clipped.append(_intersect_edge(previous, current, edge, width, height))
            clipped.append(current)
        elif previous_inside:
            clipped.append(_intersect_edge(previous, current, edge, width, height))

        previous = current
        previous_inside = current_inside

    return clipped


def _clip_polygon_to_rect(points: list, width: int, height: int) -> list:
    clipped = points
    for edge in ("left", "right", "top", "bottom"):
        clipped = _clip_against_edge(clipped, edge, width, height)
    return clipped


def transform_polygon_coords(
    points: list,
    crop_region: CropRegion
) -> list:
    """
    转换单个多边形的坐标

    Args:
        points: 原始坐标列表 [[x, y, z], ...]
        crop_region: 裁剪区域

    Returns:
        转换后的坐标列表
    """
    new_points = []
    x_offset = crop_region.x_min
    y_offset = crop_region.y_min

    for point in points:
        if len(point) >= 2:
            new_points.append([
                point[0] - x_offset,
                point[1] - y_offset,
                point[2] if len(point) > 2 else 0.0
            ])

    if len(new_points) >= 3:
        return _clip_polygon_to_rect(
            new_points,
            crop_region.width,
            crop_region.height
        )

    return [
        point for point in new_points
        if _is_inside_rect(point, crop_region.width, crop_region.height)
    ]


def transform_label_json(
    label_data: dict,
    crop_region: CropRegion
) -> dict:
    """
    转换标注 JSON 中的所有坐标

    Args:
        label_data: 原始标注 JSON 数据
        crop_region: 裁剪区域

    Returns:
        转换后的标注数据
    """
    result = copy.deepcopy(label_data)

    # 更新 FileInfo 中的尺寸
    if "FileInfo" in result:
        result["FileInfo"]["Width"] = crop_region.width
        result["FileInfo"]["Height"] = crop_region.height

    # 处理 PolygonModel2 中的多边形
    if "Models" in result and "PolygonModel2" in result["Models"]:
        if result["Models"]["PolygonModel2"] is not None:
            for polygon in result["Models"]["PolygonModel2"]:
                if "Points" in polygon and polygon["Points"] is not None:
                    polygon["Points"] = transform_polygon_coords(
                        polygon["Points"],
                        crop_region
                    )

    # 处理 Polys 中的新结构多边形
    for poly in result.get("Polys") or []:
        for shape in poly.get("Shapes") or []:
            if shape.get("Points") is None:
                continue

            source_points = []
            points_are_dicts = True
            for point in shape["Points"]:
                if isinstance(point, dict):
                    source_point = point.get("Pos")
                    if source_point is None:
                        continue
                    source_points.append(source_point)
                else:
                    points_are_dicts = False
                    source_points.append(point)

            transformed_points = transform_polygon_coords(source_points, crop_region)
            if points_are_dicts:
                shape["Points"] = [{"Pos": point} for point in transformed_points]
            else:
                shape["Points"] = transformed_points

    return result


def save_label_json(
    label_data: dict,
    output_path: Path
) -> bool:
    """
    保存标注 JSON 文件

    Args:
        label_data: 标注数据
        output_path: 输出路径

    Returns:
        是否成功
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(label_data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving label to {output_path}: {e}")
        return False


def get_polygons_from_label(label_data: dict) -> list[dict]:
    """
    从标注数据中提取所有多边形

    Args:
        label_data: 标注 JSON 数据

    Returns:
        多边形列表，每个包含 Label 和 Points
    """
    polygons = []

    if "Models" in label_data and "PolygonModel2" in label_data["Models"]:
        if label_data["Models"]["PolygonModel2"] is not None:
            for polygon in label_data["Models"]["PolygonModel2"]:
                if polygon.get("Points"):
                    polygons.append({
                        "label": polygon.get("Label"),
                        "points": polygon["Points"],
                        "closed": polygon.get("closed", False)
                    })

    for poly in label_data.get("Polys") or []:
        for shape in poly.get("Shapes") or []:
            raw_points = shape.get("Points")
            if not raw_points:
                continue

            points = []
            for point in raw_points:
                if isinstance(point, dict):
                    pos = point.get("Pos")
                    if pos is not None:
                        points.append(pos)
                else:
                    points.append(point)

            if points:
                polygons.append({
                    "label": shape.get("labelType", shape.get("Label")),
                    "points": points,
                    "closed": True
                })

    return polygons


def get_color_for_label(label_id: int, label_data: dict) -> tuple:
    """
    根据 label ID 获取对应的颜色

    Args:
        label_id: 标签 ID
        label_data: 标注数据

    Returns:
        (R, G, B) 颜色元组
    """
    default_colors = {
        1: (255, 0, 0),      # tube - 红
        2: (0, 255, 0),      # organ - 绿
        4: (255, 255, 0),    # uncertain - 黄
        5: (3, 28, 255),     # good - 蓝
        6: (255, 255, 255),  # poor - 白
        7: (85, 0, 255),     # liver - 紫
        8: (85, 0, 127),     # spleen - 深紫
        9: (170, 255, 0),    # pancreas - 黄绿
        10: (205, 133, 63),  # kidney - 棕
    }

    # 尝试从 ColorLabelTableModel 获取颜色
    if "Models" in label_data and "ColorLabelTableModel" in label_data["Models"]:
        for item in label_data["Models"]["ColorLabelTableModel"]:
            if item.get("ID") == label_id:
                color = item.get("Color", [])
                if len(color) >= 3:
                    return (color[0], color[1], color[2])

    return default_colors.get(label_id, (255, 255, 255))
