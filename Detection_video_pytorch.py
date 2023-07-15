import sys
import torch
import torchvision
import cv2
from torchvision.transforms import functional as F
from models.yolo import Model
from models.yolo import SegmentationModel

checkpoint = torch.load('/Users/mananbhatt/Downloads/yolov5/yolov5s-seg.pt')
model_dict = checkpoint['model']  # Access the model from the checkpoint dictionary

# Create an instance of the YOLOv5 model
model = models.yolo.Model()  # Replace `models.yolo` with the correct module path

# Load the weights into the model
model.load_state_dict(model_dict)
model.eval()

# Set the device to CPU or GPU (if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

video_path = 'input.mp4'        # Add input and output paths.
output_path = 'output.mp4'
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the minimum distance for social distancing (in pixels)
min_distance = 100

# Define a font for displaying text on the output video
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.6
font_thickness = 1

# Initialize variables
num_violations = 0

# Create VideoWriter object to save the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to a PyTorch tensor
    frame_tensor = F.to_tensor(frame).unsqueeze(0).to(device)

    # Perform object detection
    with torch.no_grad():
        detections = model(frame_tensor)[0]

    # Filter detections for person class
    person_detections = detections[detections[:, 5] == 0]

    # Check for violations in social distancing
    for i in range(len(person_detections)):
        for j in range(i + 1, len(person_detections)):
            x1, y1, x2, y2, conf = person_detections[i, :5]
            x3, y3, x4, y4, conf = person_detections[j, :5]
            
            # Calculate the distance between bounding box centers
            center_x1 = (x1 + x2) / 2
            center_y1 = (y1 + y2) / 2
            center_x2 = (x3 + x4) / 2
            center_y2 = (y3 + y4) / 2
            distance = ((center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2) ** 0.5
            
            # If the distance is less than the minimum distance, it's a violation
            if distance < min_distance:
                num_violations += 1
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.rectangle(frame, (int(x3), int(y3)), (int(x4), int(y4)), (0, 0, 255), 2)
                cv2.putText(frame, 'Violation', (int(x1), int(y1) - 5), font, font_scale, (0, 0, 255), font_thickness, cv2.LINE_AA)

    # Draw the number of violations on the frame
    cv2.putText(frame, f'Violations: {num_violations}', (10, 30), font, font_scale, (0, 0, 255), font_thickness, cv2.LINE_AA)

    # Write the frame to the output video
    output_video.write(frame)

    # Display the resulting frame
    cv2.imshow('Social Distancing', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer objects
cap.release()
output_video.release()

# Destroy all windows
cv2.destroyAllWindows()
