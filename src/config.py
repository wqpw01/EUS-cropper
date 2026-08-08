"""
裁剪配置模块

定义标准分辨率下的裁剪坐标，以及比例换算逻辑
"""

from dataclasses import dataclass


@dataclass
class CropRegion:
    """裁剪区域定义"""
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    @property
    def width(self) -> int:
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        return self.y_max - self.y_min

    def as_tuple(self) -> tuple:
        """返回 (left, upper, right, lower) 格式，用于 PIL Image.crop"""
        return (self.x_min, self.y_min, self.x_max, self.y_max)


# 标准 1920x1080 分辨率下的固定裁剪坐标
STANDARD_WIDTH = 1920
STANDARD_HEIGHT = 1080
STANDARD_CROP_REGION = CropRegion(
    x_min=626,
    y_min=168,
    x_max=1530,
    y_max=935
)

# 比例换算（用于非标准分辨率）
# 宽度占比：626/1920 ~ 1530/1920
# 高度占比：168/1080 ~ 935/1080
WIDTH_RATIO_MIN = 0.3260416667
WIDTH_RATIO_MAX = 0.796875
HEIGHT_RATIO_MIN = 0.1555555556
HEIGHT_RATIO_MAX = 0.8657407407


def calculate_crop_region(image_width: int, image_height: int) -> CropRegion:
    """
    根据图像分辨率计算裁剪区域

    对于标准 1920x1080 分辨率，使用固定坐标
    对于其他分辨率，按比例换算

    Args:
        image_width: 图像宽度
        image_height: 图像高度

    Returns:
        CropRegion: 计算后的裁剪区域
    """
    if image_width == STANDARD_WIDTH and image_height == STANDARD_HEIGHT:
        return STANDARD_CROP_REGION

    # 按比例计算
    x_min = int(image_width * WIDTH_RATIO_MIN)
    y_min = int(image_height * HEIGHT_RATIO_MIN)
    x_max = int(image_width * WIDTH_RATIO_MAX)
    y_max = int(image_height * HEIGHT_RATIO_MAX)

    return CropRegion(x_min, y_min, x_max, y_max)


# 默认输入输出路径配置
from pathlib import Path

# 数据源路径（WSL 下 Windows 路径）
INPUT_BASE_PATH = Path("/mnt/c/Users/zhangyutang/Desktop/CT-EUS定位项目/数据/1020真值/配对图片")

# 输出路径
CROPPED_OUTPUT_PATH = Path("/mnt/c/Users/zhangyutang/Desktop/CT-EUS定位项目/数据/1020真值/裁剪后")
OUTPUT_BASE_PATH = CROPPED_OUTPUT_PATH.parent
TEST_VIS_PATH = OUTPUT_BASE_PATH / "test_visualizations"
