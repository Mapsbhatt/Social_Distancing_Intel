import cv2
import numpy as np
from openvino.inference_engine import IECore


def preprocess_frame(frame, input_shape):
    resized_frame = cv2.resize(frame, (input_shape[3], input_shape[2]))
    processed_frame = resized_frame.transpose((2, 0, 1))
    processed_frame = np.expand_dims(processed_frame, axis=0)
    return processed_frame


def draw_bounding_boxes(frame, boxes, violation_count):
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    cv2.putText(frame, f"Violations: {violation_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return frame


def calculate_euclidean_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance


def check_social_distancing(boxes, distance_threshold):
    violation_count = 0
    centers = []
    for box in boxes:
        center_x = (box[0] + box[2]) // 2
        center_y = box[3]
        centers.append((center_x, center_y))

    for i, center1 in enumerate(centers):
        for j, center2 in enumerate(centers[i + 1:], i + 1):
            distance = calculate_euclidean_distance(center1, center2)
            if distance < distance_threshold:
                violation_count += 1
                cv2.line(frame, center1, center2, (0, 0, 255), 2)

    return violation_count


# Load the OpenVINO Inference Engine
ie = IECore()

# Load the IR model files
model_xml = "yolov5s.xml"
model_bin = "yolov5s.bin"

# Load the network
net = ie.read_network(model=model_xml, weights=model_bin)

# Get the input and output layer names
input_blob = next(iter(net.input_info))
output_blob = next(iter(net.outputs))

# Load the network to the device (CPU)
exec_net = ie.load_network(network=net, device_name="CPU")

# Define input and output shapes
input_shape = net.input_info[input_blob].input_data.shape
output_shape = net.outputs[output_blob].shape

# Define the distance threshold for social distancing
distance_threshold = 100

# Open the video file
video_path = "video.mp4"      #Add the input path.
video = cv2.VideoCapture(video_path)

# Create a VideoWriter object to save the output video
output_path = "output.avi"   #Add the output path.
output_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
output_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
video_writer = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))

while True:
    # Read a frame from the video
    ret, frame = video.read()
    if not ret:
        break

    # Preprocess the frame
    processed_frame = preprocess_frame(frame, input_shape)

    # Run inference
    outputs = exec_net.infer(inputs={input_blob: processed_frame})

    # Parse the output
    output = outputs[output_blob][0]

    # Filter detections with confidence above a threshold
    detections = output[output[:, 4] > 0.5]

    # Extract bounding boxes
    boxes = detections[:, :4].astype(int)

    # Draw bounding boxes and check social distancing
    frame = draw_bounding_boxes(frame, boxes, check_social_distancing(boxes, distance_threshold))

    # Display the frame
    cv2.imshow("Social Distance Detection", frame)
    video_writer.write(frame)

    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video.release()
video_writer.release()
cv2.destroyAllWindows()
