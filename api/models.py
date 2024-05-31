from pydantic import BaseModel


class HealthCheckModel(BaseModel):
    """Healthcheck"""
    status: bool = None


class VerifcationResponseModel(BaseModel):
    """Results of verification"""
    result: bool = None
