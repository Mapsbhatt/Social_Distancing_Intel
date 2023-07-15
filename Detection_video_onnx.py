import cv2
import numpy as np
import onnxruntime as ort
import time

# Load the YOLOv5s model in ONNX format
model_path = "yolov5s.onnx"    #Plug-in the path
session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name
output_names = [output.name for output in session.get_outputs()]

# Load video input
video_path = "input.mp4"    #Plug-in the path
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create video writer for the output
output_path = "output.mp4"    #Plug-in the path
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Define social distancing thresholds (in pixels)
min_distance = 80  # Minimum allowable distance between persons
start_time = time.time()
frame_count = 0

# Process each frame in the video
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    resized_frame = cv2.resize(frame, (640, 640))
    input_data = np.expand_dims(resized_frame.transpose(2, 0, 1), axis=0).astype(np.float32) / 255.0

    # Run inference
    outputs = session.run(output_names, {input_name: input_data})

    # Post-process output
    detections = outputs[0]
    num_detections = len(detections)

    # Draw bounding boxes and calculate violations
    violations = 0
    for i in range(num_detections):
        detection = detections[i]
        class_id = detection[0]
        confidence = detection[5]
        x, y, w, h = detection[1:5]

        # Filter out non-person classes (adjust class_id if needed)
        if int(class_id[0]) != 0:
            continue

        # Draw bounding box
        left = int((x - w / 2) * width)
        top = int((y - h / 2) * height)
        right = int((x + w / 2) * width)
        bottom = int((y + h / 2) * height)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Calculate center coordinates
        cx = (left + right) // 2
        cy = (top + bottom) // 2

        # Check violations with other persons
        for j in range(i + 1, num_detections):
            detection2 = detections[j]
            class_id2 = detection2[0]
            x2, y2, w2, h2 = detection2[1:5]

            # Filter out non-person classes (adjust class_id2 if needed)
            if int(class_id2) != 0:
                continue

            # Calculate center coordinates of the second person
            cx2 = int((x2 + w2 / 2) * width)
            cy2 = int((y2 + h2 / 2) * height)

            # Calculate Euclidean distance between centers
            distance = np.sqrt((cx - cx2) ** 2 + (cy - cy2) ** 2)

            # Check if distance violates social distancing
            if distance < min_distance:
                violations += 1
                cv2.line(frame, (cx, cy), (cx2, cy2), (0, 255, 0), 2)
 # Draw violations count on the frame
    cv2.putText(frame, f"Violations: {violations}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Calculate FPS
    frame_count += 1
    elapsed_time = time.time() - start_time
    fps_output = frame_count / elapsed_time
    cv2.putText(frame, f"FPS: {fps_output:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Write frame with bounding boxes to output video
    out.write(frame)

    # Display frame with bounding boxes (optional)
    cv2.imshow("Social Distancing Detection", frame)
    if cv2.waitKey(1) == ord("q"):
        break
        
# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
