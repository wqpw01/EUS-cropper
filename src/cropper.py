"""
图像裁剪模块
"""

from pathlib import Path
from typing import Optional
from PIL import Image

from .config import calculate_crop_region, CropRegion


def crop_image(
    image_path: Path,
    output_path: Path,
    crop_region: Optional[CropRegion] = None
) -> tuple[bool, Optional[CropRegion]]:
    """
    裁剪单张图像

    Args:
        image_path: 输入图像路径
        output_path: 输出图像路径
        crop_region: 裁剪区域，如果为 None 则自动计算

    Returns:
        (success, crop_region): 是否成功，实际使用的裁剪区域
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size

            if crop_region is None:
                crop_region = calculate_crop_region(width, height)

            # 裁剪
            cropped = img.crop(crop_region.as_tuple())

        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 保存
        cropped.save(output_path)

        return True, crop_region

    except Exception as e:
        print(f"Error cropping {image_path}: {e}")
        return False, crop_region


def crop_image_in_memory(image_path: Path) -> tuple[Optional[Image.Image], Optional[CropRegion]]:
    """
    在内存中裁剪图像（用于可视化测试）

    Args:
        image_path: 输入图像路径

    Returns:
        (cropped_image, crop_region): 裁剪后的图像和裁剪区域
    """
    crop_region = None

    try:
        with Image.open(image_path) as img:
            width, height = img.size
            crop_region = calculate_crop_region(width, height)
            cropped = img.crop(crop_region.as_tuple())
        return cropped, crop_region
    except Exception as e:
        print(f"Error cropping {image_path}: {e}")
        return None, crop_region
