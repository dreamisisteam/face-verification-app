from contextlib import asynccontextmanager

from fastapi import FastAPI

from verification_modules import models
from api import routes


@asynccontextmanager
async def verification_models_lifespan(app: FastAPI):
    """Activate preparation of used models"""
    for model in models.VERIFICATION_MODELS_INSTANCES_MAPPING.values():
        model()

    yield


app = FastAPI(
    lifespan=verification_models_lifespan,
)
app.include_router(routes.router)
