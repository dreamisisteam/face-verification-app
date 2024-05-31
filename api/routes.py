from typing import Annotated

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from verification_modules import models, detectors
from api import models as api_models

router = APIRouter(prefix="/api/v1")


@router.get("/health", response_model=api_models.HealthCheckModel)
def get_health():
    """Healthcheck"""
    return {"status": True}


@router.post("/verificate/", response_model=api_models.VerifcationResponseModel)
async def process_vefification(
    files: Annotated[list[UploadFile], File()],
    model_name: Annotated[
        models.VERIFICATION_MODELS_LITERAL, Form()
    ] = models.DEFAULT_VERIFICATION_MODEL_NAME,
    detector_name: Annotated[
        detectors.DETECTORS_LITERAL, Form()
    ] = detectors.DEFAULT_DETECTOR_NAME,
):
    """Model verification"""
    if len(files) != 2:
        raise HTTPException(
            status_code=400, detail={"files": "Should be provided exactly 2 files!"}
        )

    image_binaries = []

    for file in files:
        if "image/" not in file.content_type:
            raise HTTPException(status_code=400, detail={"files": "Should be images!"})

        image_binaries.append(await file.read())

    try:
        result, *_ = models.VERIFICATION_MODELS_INSTANCES_MAPPING[
            model_name
        ].run_verification(
            detector_name=detector_name,
            images_raw=image_binaries,
        )
    except ValueError as ex:
        raise HTTPException(status=400, detail=str(ex))
    except Exception as ex:
        raise HTTPException(status=500, detail=str(ex))

    return {
        "result": result,
    }
