import io

import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

from PIL import Image

from keras.models import Sequential, Model
from keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Activation
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input

from .base import BaseVerificationModel


class VGGFaceModel(BaseVerificationModel):
    """VGG-Face Model"""

    def prepare_model(self) -> Sequential:
        model = Sequential()

        model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
        model.add(Convolution2D(64, (3, 3), activation="relu"))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(64, (3, 3), activation="relu"))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, (3, 3), activation="relu"))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, (3, 3), activation="relu"))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, (3, 3), activation="relu"))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, (3, 3), activation="relu"))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, (3, 3), activation="relu"))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation="relu"))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation="relu"))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation="relu"))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation="relu"))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation="relu"))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation="relu"))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(Convolution2D(4096, (7, 7), activation="relu"))
        model.add(Dropout(0.5))
        model.add(Convolution2D(4096, (1, 1), activation="relu"))
        model.add(Dropout(0.5))
        model.add(Convolution2D(2622, (1, 1)))
        model.add(Flatten())
        model.add(Activation("softmax"))

        model.load_weights(self.pretrained_weights_path)

        descriptor = Model(
            inputs=model.layers[0].input, outputs=model.layers[-2].output
        )
        return descriptor

    def _get_face_representation(self, image_array: np.ndarray) -> np.ndarray:
        with io.BytesIO() as byte_io:
            image = Image.fromarray(image_array)
            image.save(byte_io, format="JPEG")

            jpg_buffer = byte_io.getvalue()

            prepared_image = load_img(io.BytesIO(jpg_buffer), target_size=(224, 224))

            prepared_image_array = img_to_array(prepared_image)
            prepared_image_array = np.expand_dims(prepared_image_array, axis=0)
            prepared_image_array = preprocess_input(prepared_image_array)

            return self.raw_model.predict(prepared_image_array)

    def _verificate(self, *representations: tuple[np.ndarray, np.ndarray]) -> bool:
        return (1 - cosine_similarity(*representations)[0][0]) < 0.4
