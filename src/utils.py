"""
工具函数模块
"""

import tarfile
from pathlib import Path
from typing import Optional

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def extract_label_json(tar_path: Path) -> Optional[dict]:
    """
    从 tar 文件中提取 JSON 标注内容

    Args:
        tar_path: tar 文件路径

    Returns:
        解析后的 JSON 字典，如果失败返回 None
    """
    import json

    try:
        with tarfile.open(tar_path, 'r') as tar:
            # tar 文件中应该只有一个 JSON 文件
            members = tar.getmembers()
            if not members:
                return None

            json_member = members[0]
            f = tar.extractfile(json_member)
            if f is None:
                return None

            content = f.read().decode('utf-8')
            return json.loads(content)
    except Exception as e:
        print(f"Error extracting {tar_path}: {e}")
        return None


def get_label_tar_path(image_path: Path) -> Optional[Path]:
    """
    根据图像路径获取对应的标注 tar 文件路径

    Args:
        image_path: 图像文件路径

    Returns:
        标注 tar 文件路径，如果不存在返回 None
    """
    # 标注文件命名格式:
    # frame_00000001.png -> frame_00000001_png_Label.tar
    # IMG_01.abc.jpg -> IMG_01_abc_jpg_Label.tar
    name = image_path.stem
    ext = image_path.suffix.lstrip('.').lower()

    label_names = [
        f"{name}_{ext}_Label.tar",
        f"{name.replace('.', '_')}_{ext}_Label.tar",
    ]

    for label_name in dict.fromkeys(label_names):
        label_path = image_path.parent / label_name
        if label_path.exists():
            return label_path
    return None


def find_image_files(image_dir: Path) -> list[Path]:
    """
    查找目录中的所有支持格式图像

    Args:
        image_dir: 图像目录

    Returns:
        排序后的图像路径列表
    """
    if not image_dir.exists() or not image_dir.is_dir():
        return []

    return sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def find_image_label_pairs(group_dir: Path) -> list[tuple[Path, Optional[Path]]]:
    """
    查找目录中的所有图像-标注对

    Args:
        group_dir: group 文件夹路径

    Returns:
        [(image_path, label_path), ...] 列表
    """
    pairs = []
    image_files = find_image_files(group_dir)

    for img_path in image_files:
        label_path = get_label_tar_path(img_path)
        pairs.append((img_path, label_path))

    return pairs


def find_all_groups(base_path: Path) -> list[Path]:
    """
    查找所有 group 文件夹

    Args:
        base_path: 数据根目录

    Returns:
        排序后的 group 文件夹路径列表
    """
    if not base_path.exists() or not base_path.is_dir():
        return []

    return [
        g for g in sorted(base_path.iterdir())
        if g.is_dir() and find_image_files(g)
    ]
