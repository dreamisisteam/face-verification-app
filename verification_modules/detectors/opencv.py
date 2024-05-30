import numpy as np
import cv2

from .base import BaseDetector


class OpenCVDetector(BaseDetector):
    """Detector MultiScale of OpenCV"""

    def _execute(
        self,
        image_array: np.ndarray,
        expected_count: int | None = None,
    ) -> list[np.ndarray] | None:
        """
        Detect faces on image

        :param image_array: image array representation
        :param expected_count: expected count of faces, defaults to None
        :raises ValueError: if expected count of faces is not equal to calculated
        :return: list of faces (cropped) images representations
        """
        gray_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        face_classifier = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # TODO: params to config?
        # TODO: increase accuracy: add detectMultiScale on eyes
        faces_rects = face_classifier.detectMultiScale(
            gray_image, 1.1, 15, minSize=(40, 40)
        )

        if expected_count and len(faces_rects) != expected_count:
            raise ValueError(
                f"expected faces count: {expected_count}\nreceived: {len(faces_rects)}"
            )
        elif not expected_count and len(faces_rects) == 0:
            return None

        image_rgb_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        faces_images_arrays = [
            image_rgb_array[y:y + w, x:x + h] for (x, y, w, h) in faces_rects
        ]

        return faces_images_arrays
