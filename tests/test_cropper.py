"""
裁剪模块单元测试
"""

import pytest
from pathlib import Path
from PIL import Image
import tempfile

from src.config import (
    STANDARD_CROP_REGION,
    STANDARD_WIDTH,
    STANDARD_HEIGHT,
    calculate_crop_region
)
from src.cropper import crop_image


class TestCropRegion:
    """裁剪区域计算测试"""

    def test_standard_resolution(self):
        """标准 1920x1080 分辨率使用固定坐标"""
        region = calculate_crop_region(STANDARD_WIDTH, STANDARD_HEIGHT)

        assert region.as_tuple() == (626, 168, 1530, 935)
        assert region.x_min == STANDARD_CROP_REGION.x_min
        assert region.y_min == STANDARD_CROP_REGION.y_min
        assert region.x_max == STANDARD_CROP_REGION.x_max
        assert region.y_max == STANDARD_CROP_REGION.y_max

    def test_non_standard_resolution(self):
        """非标准分辨率按比例计算"""
        # 测试 1280x720
        region = calculate_crop_region(1280, 720)

        # 宽度约 32.6%~79.7%: 417 ~ 1020
        assert 412 <= region.x_min <= 422
        assert 1015 <= region.x_max <= 1025

        # 高度约 15.6%~86.6%: 112 ~ 623
        assert 107 <= region.y_min <= 117
        assert 618 <= region.y_max <= 628

    def test_output_dimensions(self):
        """验证输出尺寸"""
        region = calculate_crop_region(STANDARD_WIDTH, STANDARD_HEIGHT)

        # 输出尺寸: (1530-626) x (935-168) = 904 x 767
        assert region.width == 904
        assert region.height == 767


class TestCropImage:
    """图像裁剪测试"""

    def test_crop_standard_image(self):
        """测试裁剪标准分辨率图像"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建测试图像
            img_path = Path(tmpdir) / "test.png"
            output_path = Path(tmpdir) / "test_cropped.png"

            # 创建 1920x1080 纯色图像
            img = Image.new('RGB', (1920, 1080), color='red')
            img.save(img_path)

            # 裁剪
            success, region = crop_image(img_path, output_path)

            assert success is True
            assert output_path.exists()

            # 验证输出尺寸
            with Image.open(output_path) as cropped:
                assert region == STANDARD_CROP_REGION
                assert cropped.size == (region.width, region.height)

    def test_crop_with_existing_output_dir(self):
        """测试输出目录已存在的情况"""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.png"
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()
            output_path = output_dir / "test_cropped.png"

            img = Image.new('RGB', (1920, 1080), color='blue')
            img.save(img_path)

            success, _ = crop_image(img_path, output_path)

            assert success is True
            assert output_path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
