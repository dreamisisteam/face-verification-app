# face-verification-app

App to make experiments on face verification tasks.

## Pre-requirements

Get pretrained models weights:
```
# VGG-Face
https://www.kaggle.com/datasets/acharyarupak391/vggfaceweights/
```

## Package

### Description

Package is created to provide API for face verification task.

#### Supported Models
VGG-Face

#### Supported Detectors
OpenCV (cv2)

### Tests

```
pytest tests/verification_modules
```

### Build

We use pyproject.toml as config of package.

To build package you need to install "build" library:
```
make build
```

### Install

To install package from PyPI (will be available soon):
```
pip install face-verification-modules
```

## Web application

To start web application use:
```
make run_api
```

If you want to turn on developer mode:
```
fastapi dev api/app.py
```
