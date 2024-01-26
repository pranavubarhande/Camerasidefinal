import cv2
import numpy as np

# Load your pre-trained model
def load_object_detection_model():
    prototxt_path = "MobileNetSSD_deploy.prototxt"
    model_path = "MobileNetSSD_deploy.caffemodel"
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    return net

# Object detection pipeline
def object_detection_pipeline(frame, model):
    if frame is not None and not frame.size == (0, 0):
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

        model.setInput(blob)
        detections = model.forward()

        for key in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, key, 2]

            if confidence > 0.2:
                idx = int(detections[0, 0, key, 1])

                (h, w) = frame.shape[:2]
                box = detections[0, 0, key, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Print label, confidence, and coordinates
                label = "Class: {}, Confidence: {:.2f}%".format(idx, confidence * 100)
                coordinates = "Coordinates: ({}, {}, {}, {})".format(startX, startY, endX, endY)
                # print(label, coordinates)

                color = (0, 255, 0)  # Green color
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame
