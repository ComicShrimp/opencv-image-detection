import cv2


def __convert_greyscale_to_binary(image_in_greyscale):
    ret, image_in_binary = cv2.threshold(image_in_greyscale, 100, 255, cv2.THRESH_OTSU)
    return image_in_binary


def __invert_image(image_in_binary):
    return ~image_in_binary


def __find_contours(image, image_in_binary):
    contours, hierarchy = cv2.findContours(
        image_in_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    image_with_contours = cv2.drawContours(image, contours, -1, (0, 255, 0), 3)


def image_segmentation(image):

    image_for_segmentation = image.copy()

    image_in_greyscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image_in_binary = __convert_greyscale_to_binary(image_in_greyscale)
    image_in_binary = __invert_image(image_in_binary)

    __find_contours(image_for_segmentation, image_in_binary)

    return image_for_segmentation
