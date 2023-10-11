import numpy as np
import cv2
from moviepy.editor import VideoFileClip
import tensorflow as tf
from tensorflow import keras

# Load your pre-trained model
model = keras.models.load_model('model.h5')

class Lanes():
    def __init__(self):
        self.recent_fit = []
        self.avg_fit = []

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
# CLASSES = ["bicycle",
#     "bus", "car", "cow",
#     "dog", "horse", "motorbike", "person", "sheep",
#     ]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

prototxt_path = "MobileNetSSD_deploy.prototxt"
model_path = "MobileNetSSD_deploy.caffemodel"

def serializing_model():
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    return net
net = serializing_model()

def detect_objects(frame):

    if frame is not None and not frame.size == (0, 0):
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
        0.007843, (300, 300), 127.5)

    net.setInput(blob)
    detections = net.forward()

    for key in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, key, 2]

        if confidence > 0.2:  
            idx = int(detections[0, 0, key, 1])

            (h, w) = frame.shape[:2]
            box = detections[0, 0, key, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            color = COLORS[idx]
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            if(confidence > 0.8):
                cv2.putText(frame, "Alert: Objects around:" + CLASSES[idx], (570, 560), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
            

    return frame
def road_lines(image):
    small_img = cv2.resize(image, (160, 80))  # Resize image using OpenCV
    small_img = np.array(small_img)
    small_img = small_img[None,:,:,:]

    prediction = model.predict(small_img)[0] * 255

    lanes.recent_fit.append(prediction)
   
    if len(lanes.recent_fit) > 5:
        lanes.recent_fit = lanes.recent_fit[1:]

    lanes.avg_fit = np.mean(np.array([i for i in lanes.recent_fit]), axis=0)
    
    blanks = np.zeros_like(lanes.avg_fit).astype(np.uint8)
    lane_drawn = np.dstack((blanks, lanes.avg_fit, blanks))

    lane_image = cv2.resize(lane_drawn, (1280, 720))  # Resize the lane image

    # Convert both images to np.uint8
    image = image.astype(np.uint8)
    lane_image = lane_image.astype(np.uint8)

    # Apply addWeighted with output data type specified as np.uint8
    result = cv2.addWeighted(image, 1, lane_image, 1, 0, dtype=cv2.CV_8U)
    result = detect_objects(result)

    return result


lanes = Lanes()

# If you want to process a video, uncomment the following lines
def process_video(output_video_path):
    cap = cv2.VideoCapture(0)
    frame_skip = 5  # Process every third frame
    frame_count = 0

    # Get video properties
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(5))

    # Define the codec and create a VideoWriter object for the output video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Initialize variables for running average and frame index
    running_avg = None
    index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Break the loop when the video ends

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        # Process the frame using the vid_pipeline function
        # frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        frame = cv2.resize(frame, (1280, 720))
        processed_frame = road_lines(frame)
        # processed_frame = colorit(processed_frame)

        # Write the processed frame to the output video
        # out.write(processed_frame[0])

        # Display the processed frame (optional)
        cv2.imshow('Processed Video', processed_frame)

        # Press 'q' to exit the video playback
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()

    cv2.destroyAllWindows()

process_video('newvideo.mp4')


