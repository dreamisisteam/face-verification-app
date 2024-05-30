import abc
from typing import Any
from pathlib import Path

import numpy as np

from verification_modules import utils, detectors


class BaseVerificationModel(abc.ABC):
    """Base class of Verification Model"""

    pretrained_weights_path: Path = "data/vgg_face_weights.h5"

    _model: Any = None

    def __init__(
        self,
        pretrained_weights_path: str | Path | None = None,
    ) -> None:
        self.pretrained_weights_path = Path(
            pretrained_weights_path or self.pretrained_weights_path
        )

    def __call__(self):
        self._get_model()

    @abc.abstractmethod
    def prepare_model(self):
        """Prepare model to executions"""
        raise NotImplementedError("prepare_model")

    def _get_model(self):
        model = self.prepare_model()
        self._model = model
        return model

    @property
    def raw_model(self):
        """Raw instance of model"""
        if model := self._model:
            return model
        return self._get_model()

    @abc.abstractmethod
    def _get_face_representation(self, image_array: np.ndarray) -> Any:
        raise NotImplementedError("_get_face_representation")

    @abc.abstractmethod
    def _verificate(self, *representations: tuple[Any, Any]) -> bool:
        raise NotImplementedError("_verificate")

    def run_verification(
        self,
        detector_name: detectors.DETECTORS_LITERAL | None = None,
        images_pathes: tuple[str, str] | tuple[str] = None,
        images_raw: tuple[bytes, bytes] | tuple[bytes] = None,
    ) -> tuple[bool, list[np.ndarray]]:
        """
        Run verification process on 2 images
        :param detector_name: name of used detector
        :param images_pathes: paths to images, defaults to None
        :param images_raw: binaries images, defaults to None
        :return: result of verification, arrays of faces (cropped images)
        """
        detector = detectors.DETECTORS_MAPPING[detector_name]

        images_pathes = images_pathes or []
        images_raw = images_raw or []

        images_arrays = utils.get_images_arrays(*images_pathes, *images_raw)

        faces_images_arrays = [
            detector(_image_array, expected_count=1)[0]
            for _image_array in images_arrays
        ]
        faces_representations = [
            self._get_face_representation(_face_image_array)
            for _face_image_array in faces_images_arrays
        ]

        return self._verificate(*faces_representations), faces_images_arrays
