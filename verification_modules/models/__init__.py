from typing import Literal

from .vgg_face import VGGFaceModel

VERIFICATION_MODELS_MAPPING = {
    "vgg_face": VGGFaceModel,
}
# TODO: get pretrained paths from config?
VERIFICATION_MODELS_INSTANCES_MAPPING = {
    "vgg_face": VGGFaceModel(),
}
VERIFICATION_MODELS_LITERAL = Literal["vgg_face"]

DEFAULT_VERIFICATION_MODEL_NAME = "vgg_face"
DEFAULT_VERIFICATION_MODEL = VERIFICATION_MODELS_MAPPING[
    DEFAULT_VERIFICATION_MODEL_NAME
]
