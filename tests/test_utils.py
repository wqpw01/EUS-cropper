"""
工具函数单元测试
"""

from pathlib import Path

from PIL import Image

from src.utils import find_image_files, get_label_tar_path


def test_get_label_tar_path_supports_dotted_jpg_names(tmp_path):
    """测试点号 jpg 文件名能匹配下划线 tar 标签名"""
    image_path = tmp_path / "IMG_01.abc.0001.jpg"
    label_path = tmp_path / "IMG_01_abc_0001_jpg_Label.tar"
    image_path.write_bytes(b"fake image")
    label_path.write_bytes(b"fake tar")

    assert get_label_tar_path(image_path) == label_path


def test_find_image_files_supports_png_and_jpg(tmp_path):
    """测试图像发现支持 png/jpg/jpeg"""
    names = ["a.png", "b.jpg", "c.jpeg", "d.txt"]
    for name in names:
        path = tmp_path / name
        if path.suffix in {".png", ".jpg", ".jpeg"}:
            Image.new("RGB", (10, 10)).save(path)
        else:
            path.write_text("ignore", encoding="utf-8")

    found = find_image_files(tmp_path)

    assert [p.name for p in found] == ["a.png", "b.jpg", "c.jpeg"]
