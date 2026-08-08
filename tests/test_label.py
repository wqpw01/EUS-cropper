"""
标注处理模块单元测试
"""

import pytest
import json

from src.config import CropRegion
from src.label_processor import (
    transform_polygon_coords,
    transform_label_json,
    get_polygons_from_label,
    get_color_for_label
)


class TestTransformPolygonCoords:
    """多边形坐标转换测试"""

    def test_basic_transform(self):
        """基本坐标偏移测试"""
        points = [
            [600, 200, 0.0],
            [800, 300, 0.0],
            [1000, 400, 0.0]
        ]

        crop_region = CropRegion(576, 118, 1580, 985)
        result = transform_polygon_coords(points, crop_region)

        assert len(result) == 3
        # 600 - 576 = 24, 200 - 118 = 82
        assert result[0] == [24, 82, 0.0]
        # 800 - 576 = 224, 300 - 118 = 182
        assert result[1] == [224, 182, 0.0]
        # 1000 - 576 = 424, 400 - 118 = 282
        assert result[2] == [424, 282, 0.0]

    def test_filter_out_of_bounds(self):
        """测试严格裁剪超出裁剪区域的多边形并用边缘补全"""
        points = [
            [50, 150, 0.0],     # 左侧区域外
            [200, 150, 0.0],    # 区域内
            [200, 250, 0.0],    # 区域内
            [50, 250, 0.0],     # 左侧区域外
        ]

        crop_region = CropRegion(100, 100, 300, 300)
        result = transform_polygon_coords(points, crop_region)

        assert result == [
            [0.0, 50.0, 0.0],
            [100, 50, 0.0],
            [100, 150, 0.0],
            [0.0, 150.0, 0.0],
        ]

    def test_empty_points(self):
        """空点列表测试"""
        crop_region = CropRegion(576, 118, 1580, 985)
        result = transform_polygon_coords([], crop_region)
        assert result == []


class TestTransformLabelJson:
    """标注 JSON 转换测试"""

    def test_updates_file_info(self):
        """测试更新 FileInfo 中的尺寸"""
        label_data = {
            "FileInfo": {
                "Width": 1920,
                "Height": 1080
            }
        }

        crop_region = CropRegion(576, 118, 1580, 985)
        result = transform_label_json(label_data, crop_region)

        assert result["FileInfo"]["Width"] == 1004
        assert result["FileInfo"]["Height"] == 867

    def test_transforms_polygons(self):
        """测试转换多边形坐标"""
        label_data = {
            "Models": {
                "PolygonModel2": [
                    {
                        "Label": 1,
                        "Points": [
                            [600, 200, 0.0],
                            [800, 300, 0.0]
                        ]
                    }
                ]
            }
        }

        crop_region = CropRegion(576, 118, 1580, 985)
        result = transform_label_json(label_data, crop_region)

        points = result["Models"]["PolygonModel2"][0]["Points"]
        assert points[0] == [24, 82, 0.0]
        assert points[1] == [224, 182, 0.0]

    def test_handles_null_polygon(self):
        """测试处理 null 多边形"""
        label_data = {
            "Models": {
                "PolygonModel2": None
            }
        }

        crop_region = CropRegion(576, 118, 1580, 985)
        result = transform_label_json(label_data, crop_region)

        assert result["Models"]["PolygonModel2"] is None

    def test_transforms_polys_points(self):
        """测试转换 Polys 结构中的多边形坐标"""
        label_data = {
            "FileInfo": {
                "Width": 1920,
                "Height": 1080
            },
            "Polys": [
                {
                    "Shapes": [
                        {
                            "labelType": 5,
                            "Points": [
                                {"Pos": [50, 150, 0.0]},
                                {"Pos": [200, 150, 0.0]},
                                {"Pos": [200, 250, 0.0]},
                                {"Pos": [50, 250, 0.0]},
                            ],
                        }
                    ]
                }
            ],
        }

        crop_region = CropRegion(100, 100, 300, 300)
        result = transform_label_json(label_data, crop_region)

        points = result["Polys"][0]["Shapes"][0]["Points"]
        assert [p["Pos"] for p in points] == [
            [0.0, 50.0, 0.0],
            [100, 50, 0.0],
            [100, 150, 0.0],
            [0.0, 150.0, 0.0],
        ]


class TestGetPolygonsFromLabel:
    """提取多边形测试"""

    def test_extracts_polygons(self):
        """测试提取多边形"""
        label_data = {
            "Models": {
                "PolygonModel2": [
                    {
                        "Label": 1,
                        "Points": [[100, 200, 0.0]],
                        "closed": False
                    },
                    {
                        "Label": 2,
                        "Points": [[300, 400, 0.0]],
                        "closed": True
                    }
                ]
            }
        }

        polygons = get_polygons_from_label(label_data)

        assert len(polygons) == 2
        assert polygons[0]["label"] == 1
        assert polygons[1]["label"] == 2

    def test_handles_empty_polygons(self):
        """测试处理空多边形"""
        label_data = {
            "Models": {
                "PolygonModel2": None
            }
        }

        polygons = get_polygons_from_label(label_data)
        assert polygons == []


class TestGetColorForLabel:
    """获取标签颜色测试"""

    def test_default_colors(self):
        """测试默认颜色映射"""
        # tube (1) 应该是红色
        color = get_color_for_label(1, {})
        assert color == (255, 0, 0)

        # organ (2) 应该是绿色
        color = get_color_for_label(2, {})
        assert color == (0, 255, 0)

    def test_unknown_label(self):
        """测试未知标签返回白色"""
        color = get_color_for_label(999, {})
        assert color == (255, 255, 255)

    def test_custom_color_from_label(self):
        """测试从标注数据获取自定义颜色"""
        label_data = {
            "Models": {
                "ColorLabelTableModel": [
                    {"ID": 1, "Color": [100, 150, 200, 255]}
                ]
            }
        }

        color = get_color_for_label(1, label_data)
        assert color == (100, 150, 200)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
