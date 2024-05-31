import argparse

from verification_modules.detectors import DETECTORS_MAPPING, DEFAULT_DETECTOR_NAME
from verification_modules.models import (
    VERIFICATION_MODELS_MAPPING,
    DEFAULT_VERIFICATION_MODEL_NAME,
)

arg_parser = argparse.ArgumentParser(
    description="Face Verification Models",
    formatter_class=argparse.RawTextHelpFormatter,
)

arg_parser.add_argument(
    "path_1",
    help="Specifies path to verification file 1",
)
arg_parser.add_argument(
    "path_2",
    help="Specifies path to verification file 2",
)

arg_parser.add_argument(
    "-d",
    "--detector",
    dest="detector_name",
    default=DEFAULT_DETECTOR_NAME,
    help=(
        f"Specifies which detector to use.\nChoices: {DETECTORS_MAPPING.keys()}\n"
        f"Default: {DEFAULT_DETECTOR_NAME}"
    ),
)
arg_parser.add_argument(
    "-m",
    "--model",
    dest="model_name",
    default=DEFAULT_VERIFICATION_MODEL_NAME,
    help=(
        f"Specifies which model to use.\nChoices: {VERIFICATION_MODELS_MAPPING.keys()}\n"
        f"Default: {DEFAULT_VERIFICATION_MODEL_NAME}"
    ),
)
arg_parser.add_argument(
    "-p",
    "--pretrained-file",
    dest="pretrained_file_path",
    default=None,
    help="Specifies path to pretrained model file weights.",
)


def main():
    args = arg_parser.parse_args()
    model = VERIFICATION_MODELS_MAPPING[args.model_name](args.pretrained_file_path)

    model()

    result = model.run_verification(
        detector_name=args.detector_name,
        images_pathes=(args.path_1, args.path_2),
    )[0]

    print("same" if result else "not same")


if __name__ == "__main__":
    main()
