from pathlib import Path

import pytest
import numpy as np

from verification_modules import utils


def test_get_images_arrays_from_paths(image_dir: Path):
    """Test util function returning np.ndarray from paths"""
    image_1_path: Path = image_dir / "test_image_1.jpg"
    image_2_path: str = str(image_dir / "test_image_2.jpeg")

    images_arrays = utils.get_images_arrays_from_paths(image_1_path, image_2_path)

    for image_array in images_arrays:
        assert isinstance(image_array, np.ndarray)


def test_get_images_arrays_from_bytes(image_dir: Path):
    """Test util function returing np.ndarray from binaries"""
    contents = []

    for filename in ("test_image_1.jpg", "test_image_2.jpeg"):
        with open(image_dir / filename, mode="rb") as file:
            contents.append(file.read())

    assert contents
    images_arrays = utils.get_images_arrays_from_binaries(*contents)

    for image_array in images_arrays:
        assert isinstance(image_array, np.ndarray)


def test_get_images_arrays(image_dir: Path):
    """Test util function returning np.ndarray from paths or binaries"""
    with open(image_dir / "test_image_1.jpg", mode="rb") as file:
        content = file.read()
    assert content

    images_arrays = utils.get_images_arrays(content, image_dir / "test_image_2.jpeg")
    for image_array in images_arrays:
        assert isinstance(image_array, np.ndarray)


def test_get_images_arrays__raises_exception_on_undefined_type():
    """Test util function raises exception on undefined data type"""
    with pytest.raises(ValueError) as exc:
        utils.get_images_arrays(3)
    assert str(exc.value) == "3: undefined type for processing"
