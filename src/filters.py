import cv2
import numpy as np


def sepia(image):
    image_with_sepia = image.copy()
    image_with_sepia = cv2.cvtColor(image_with_sepia, cv2.COLOR_BGR2RGB)
    image_with_sepia = np.array(image_with_sepia, dtype=np.float64)
    image_with_sepia = cv2.transform(
        image_with_sepia,
        np.matrix(
            [[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]]
        ),
    )
    image_with_sepia[
        np.where(image_with_sepia > 255)
    ] = 255  # clipping values greater than 255 to 255
    image_with_sepia = np.array(image_with_sepia, dtype=np.uint8)
    image_with_sepia = cv2.cvtColor(image_with_sepia, cv2.COLOR_RGB2BGR)

    return image_with_sepia


def greyscale(image):
    image_in_greyscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return cv2.merge([image_in_greyscale, image_in_greyscale, image_in_greyscale])
