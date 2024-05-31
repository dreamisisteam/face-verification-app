import io
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

import cv2


def get_images_arrays_from_paths(
    *images_paths: tuple[str | Path, ...],
) -> list[np.ndarray]:
    """
    Generate arrays from images paths

    :param images_paths: str or pathlib.Path paths to images
    :return: list of images arrays representations
    """
    return [cv2.imread(str(image_path)) for image_path in images_paths]


def get_images_arrays_from_binaries(
    *binary_images: tuple[bytes, ...],
) -> list[np.ndarray]:
    """
    Generate arrays from images binaries

    :param binary_images: bytes representations of images
    :return: list of images arrays representations
    """
    return [
        np.array(Image.open(io.BytesIO(binary_image))) for binary_image in binary_images
    ]


def get_images_arrays(
    *images_representasions: tuple[str | Path | bytes, ...],
    raise_exception: bool = True,
) -> list[np.ndarray]:
    """
    Generate arrays from binaries or paths

    :param images_representations: representations of images
    :return: tuple of images arrays representations
    """
    images_arrays = []

    for image_representation in images_representasions:
        if isinstance(image_representation, bytes):
            images_arrays.extend(get_images_arrays_from_binaries(image_representation))
        elif isinstance(image_representation, str) or isinstance(
            image_representation, Path
        ):
            images_arrays.extend(get_images_arrays_from_paths(image_representation))
        elif raise_exception:
            raise ValueError(
                f"{repr(image_representation)}: undefined type for processing"
            )

    return images_arrays


def display_image(image_array: np.ndarray, size: tuple[int, int] = None) -> None:
    """
    Display image
    :param image_array: array representation of image
    :param size: size of window, defaults to None
    """
    plt.figure(figsize=size or (3, 3))
    plt.imshow(image_array)
    plt.axis("off")
