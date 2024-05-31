from pathlib import Path

import pytest
from httpx import AsyncClient


async def test_get_health(client: AsyncClient):
    """ GET /api/v1/health """
    response = await client.get('/api/v1/health')

    assert response.status_code == 200
    assert response.json() == {'status': True}


async def test_process_verification(
        mock_model_verification, client: AsyncClient, image_dir: Path):
    """ POST /api/v1/verificate/ """
    files = _get_files(image_dir)

    response = await client.post('/api/v1/verificate/', files=files)

    assert response.status_code == 200
    assert response.json() == {
        'result': True,
    }

    # check that verification was called only once
    mock_model_verification.assert_called_once()


@pytest.mark.parametrize('data', ({'model_name': 'dll'}, {'detector_name': 'yolo'}))
async def test_process_verification__invalid_choices(
        mock_model_verification, client: AsyncClient, image_dir: Path,
        data: dict):
    """ POST /api/v1/verificate/ - invalid choices in form """
    files = _get_files(image_dir)

    response = await client.post('/api/v1/verificate/', files=files, data=data)
    assert response.status_code == 422

    # check that after validation verification was not called
    mock_model_verification.assert_not_called()


@pytest.mark.parametrize('images_names', (
        ('test_image_1.jpg',),
        ('test_image_1.jpg', 'test_image_2.jpeg', 'black.jpeg'),
))
async def test_process_verification__invalid_files_count(
        mock_model_verification, client: AsyncClient, image_dir: Path,
        images_names: tuple[str]):
    """ POST /api/v1/verificate/ - invalid files count """
    files = _get_files(image_dir, images_names)

    response = await client.post('/api/v1/verificate/', files=files)

    assert response.status_code == 400
    assert response.json() == {
        'detail': {'files': 'Should be provided exactly 2 files!'}
    }

    # check that after validation verification was not called
    mock_model_verification.assert_not_called()


async def test_process_verifiaction__invalid_file_format(
        mock_model_verification, client: AsyncClient, image_dir: Path):
    """ POST /api/v1/verificate/ - invalid files extensions """
    files = _get_files(image_dir, ('black.jpeg', 'a.txt'))

    response = await client.post('/api/v1/verificate/', files=files)

    assert response.status_code == 400
    assert response.json() == {
        'detail': {'files': 'Should be images!'}
    }

    # check that after validation verification was not called
    mock_model_verification.assert_not_called()


@pytest.mark.parametrize('error, expected_status_code', (
        (ValueError, 400), (KeyError, 500),
))
async def test_process_verification__raised_error(
        mocker, client: AsyncClient, image_dir: Path,
        error: Exception, expected_status_code: int,
):
    """ POST /api/v1/verificate/ - error on processing """
    files = _get_files(image_dir)

    def _side_effect_error(*args, **kwargs):
        raise error()

    mocker.patch(
        'verification_modules.models.vgg_face.VGGFaceModel.prepare_model',
        return_value=None,
    )
    validation_mocker = mocker.patch(
        'verification_modules.models.base.BaseVerificationModel.run_verification',
        side_effect=_side_effect_error,
    )

    response = await client.post('/api/v1/verificate/', files=files)
    assert response.status_code == expected_status_code

    validation_mocker.assert_called_once()


def _get_files(image_dir: Path, images_names: tuple[str] = None):
    iamges_names = images_names or ('test_image_1.jpg', 'test_image_2.jpeg')
    return [('files', open(image_dir / image_name, 'rb'))
            for image_name in iamges_names]
