from pathlib import Path

import pytest

from verification_modules import detectors, utils


@pytest.mark.parametrize("expected_count", (None, 0, 1, 2))
def test_opencv_detector(image_dir: Path, expected_count):
    """Base check for opencv detector"""
    black_image_array, face_image_array = utils.get_images_arrays_from_paths(
        image_dir / "black.jpeg",
        image_dir / "test_image_1.jpg",
    )

    detector = detectors.DETECTORS_MAPPING["opencv"]
    assert isinstance(detector, detectors.OpenCVDetector)

    if not expected_count:
        assert detector(black_image_array) is None
        assert len(detector(face_image_array, expected_count)) == 1
    elif expected_count == 1:
        assert len(detector(face_image_array, expected_count)) == 1

        with pytest.raises(ValueError) as exc:
            detector(black_image_array, expected_count)
        assert str(exc.value) == "expected faces count: 1\nreceived: 0"
    else:
        with pytest.raises(ValueError) as exc:
            detector(face_image_array, expected_count)
        assert str(exc.value) == (
            f"expected faces count: {expected_count}\n" "received: 1"
        )

        with pytest.raises(ValueError) as exc:
            detector(black_image_array, expected_count)
        assert str(exc.value) == (
            f"expected faces count: {expected_count}\n" "received: 0"
        )
