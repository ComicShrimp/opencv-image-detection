import cv2

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

    detection_model.setInputSize(360, 360)
    detection_model.setInputScale(1.0 / 120)
    detection_model.setInputMean((120, 120, 120))
    detection_model.setInputSwapRB(True)

    return detection_model


def detect_objects_in_image(image):
    class_ids, configurations, boundary_box = detection_model.detect(
        image, confThreshold=confidence_threshold
    )

    if len(class_ids) != 0:
        for classId, confidence, object_box in zip(
            class_ids.flatten(), configurations.flatten(), boundary_box
        ):
            cv2.rectangle(image, object_box, color=rgb_color, thickness=2)
            cv2.putText(
                image,
                f"%s - %s"
                % (class_names[classId - 1].upper(), round(confidence * 100, 2)),
                (object_box[0] + 10, object_box[1] + 20),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                1,
                rgb_color,
                2,
            )


def show_images(image, image_in_greyscale):
    cv2.imshow("Full RGB", image)
    cv2.imshow("GryScale", image_in_greyscale)


def exit_key_pressed():
    return not cv2.waitKey(1) & 0xFF == ord("q")


def get_webcam_image():
    success, image = webcam_video.read()
    image_in_greyscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, image_in_greyscale


# Main Program

webcam_video = cv2.VideoCapture(0)

class_names = []
with open(classFile, "rt") as f:
    class_names = f.read().rstrip("\n").split("\n")


detection_model = create_detection_model()

while exit_key_pressed():
    image, image_in_greyscale = get_webcam_image()

    detect_objects_in_image(image=image)

    show_images(image=image, image_in_greyscale=image_in_greyscale)
