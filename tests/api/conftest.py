from httpx import AsyncClient
import pytest

from api.app import app


@pytest.fixture
async def client():
    """ API Client """
    async with AsyncClient(app=app, base_url='http://testserver') as client:
        yield client


@pytest.fixture
def mock_model_verification(mocker):
    """ Model Verification """
    mocker.patch(
        'verification_modules.models.vgg_face.VGGFaceModel.prepare_model',
        return_value=None,
    )

    validation_mocker = mocker.patch(
        'verification_modules.models.base.BaseVerificationModel.run_verification',
        return_value=(True, None),
    )
    yield validation_mocker
