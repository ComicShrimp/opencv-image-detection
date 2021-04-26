import cv2
from segmentation import image_segmentation
from filters import sepia, greyscale

# Global Config

confidence_threshold = 0.55
rgb_color = (255, 34, 15)


# Files

configPath = "src/lib/trained_config.pbtxt"
weightsPath = "src/lib/trained_model/frozen_inference_graph.pb"
classFile = "src/lib/coco.names"

# Functions


def create_detection_model():
    detection_model = cv2.dnn_DetectionModel(weightsPath, configPath)

    detection_model.setInputSize(320, 320)
    detection_model.setInputScale(1.0 / 120)
    detection_model.setInputMean((120, 120, 120))
    detection_model.setInputSwapRB(True)

    return detection_model


def detect_objects_in_image(image):
    image_for_objects = image.copy()

    class_ids, configurations, boundary_box = detection_model.detect(
        image_for_objects, confThreshold=confidence_threshold
    )

    if len(class_ids) != 0:
        for classId, confidence, object_box in zip(
            class_ids.flatten(), configurations.flatten(), boundary_box
        ):
            cv2.rectangle(image_for_objects, object_box, color=rgb_color, thickness=2)
            cv2.putText(
                image_for_objects,
                f"%s - %s"
                % (class_names[classId - 1].upper(), round(confidence * 100, 2)),
                (object_box[0] + 10, object_box[1] + 20),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                1,
                rgb_color,
                2,
            )
    return image_for_objects


def show_images(
    image_with_objects, image_in_greyscale, image_in_filter, image_segmentated
):
    cv2.imshow("Full RGB", image_with_objects)
    cv2.imshow("GreyScale", image_in_greyscale)
    cv2.imshow("Filter", image_in_filter)
    cv2.imshow("Image Segmentaded", image_segmentated)


def exit_key_pressed():
    return not cv2.waitKey(1) & 0xFF == ord("q")


def get_webcam_image():
    success, image = webcam_video.read()

    return image


# Main Program

webcam_video = cv2.VideoCapture(0)

class_names = []
with open(classFile, "rt") as f:
    class_names = f.read().rstrip("\n").split("\n")


detection_model = create_detection_model()

while exit_key_pressed():
    image = get_webcam_image()

    image_with_objects = detect_objects_in_image(image)
    image_in_filter_grey = detect_objects_in_image(greyscale(image))
    image_in_filter_sepia = detect_objects_in_image(sepia(image))
    image_for_segmentation = image_segmentation(image)

    show_images(
        image_with_objects=image_with_objects,
        image_in_greyscale=image,
        image_in_filter=image_in_filter_sepia,
        image_segmentated=image_for_segmentation,
    )

webcam_video.release()
cv2.destroyAllWindows()
