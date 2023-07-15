# Import OpenVINO modules
import openvino
from openvino.inference_engine import IECore, IENetwork
import openvino.runtime as ov
import time

# Load the IR model files
model_xml = "yolov5s.xml"
model_bin = "yolov5s.bin"

ie = IECore()
net = ie.read_network(model=model_xml, weights=model_bin)

# Get the input and output layer names
input_blob = next(iter(net.input_info))
output_blob = next(iter(net.outputs)) 

print(net.outputs.keys())

# Load the network to the device (CPU, GPU, etc.)
core = ov.Core()
model = core.compile_model(model_xml,"CPU")



# Define a function to compute the Euclidean distance between two points
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# Define a threshold for the minimum distance between people
distance_threshold = 200 # pixels

# Read and preprocess the input video
import cv2
import numpy as np

#give the input video
video = cv2.VideoCapture("video.mp4")
_,frame = video.read()
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"MJPG")

#give the output address to store the video 
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter('output.mp4', fourcc, 30, (width, height), True)       #Add the output path

# used to record the time when we processed last frame
prev_frame_time = 0

# used to record the time at which we processed current frame
new_frame_time = 0
 
while True:
    # Read a frame from the video.
    ret, frame = video.read()
    
    if not ret:
        break # Exit the loop if end of video or error
    height, width = frame.shape[:2]
   
    # Preprocess the frame
    image = cv2.resize(frame, (512,512))
    image = image.transpose((2, 0, 1)) # Change data layout from HWC to CHW
    
    

    # Run inference and get the output
    infer_request = model.create_infer_request()
    input_shape = [1,3,640,640]   
    input_tensor= ov.Tensor(image.astype(np.float32))
    input_tensor.shape = input_shape
    infer_request.set_tensor(input_blob,input_tensor)
    infer_request.infer()
    output_tensor = infer_request.get_tensor(output_blob)
    output = output_tensor.data
    
    

    # Parse the output and get the bounding boxes of detected people
    boxes = []
    confidences = []
    class_ids = []
    
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    fps = str(fps)
    
    boxes = []
    confidences = []
    class_ids = []

    output_array = output  # No need for accessing a dictionary key

    for detection in output_array:
        detection_list = detection.tolist() 
        print(detection)  # Add this line to inspect each detection

        # Each detection has the format [image_id, label, conf, x_min, y_min, x_max, y_max]
        if np.any(detection[2] > 0.5): # Only keep detections with confidence > 0.5
            class_id = int(detection_list[1][0])
            if class_id == 0: # Only keep detections with label 0 (person)
                x_min = int(detection[3] * width)
                y_min = int(detection[4] * height)
                x_max = int(detection[5] * width)
                y_max = int(detection[6] * height)
                boxes.append([x_min, y_min, x_max, y_max])
                confidences.append(float(detection[2]))
                class_ids.append(class_id)
    
    n = len(boxes)
    # Apply non-maximum suppression to eliminate redundant overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    # Loop over the indices of the remaining boxes
    v = 0
    for i in indices:
        box = boxes[i]
        # Draw a bounding box around the person
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        # Get the center point of the box
        center_a = np.array([box[0] + (box[2] - box[0]) / 2, box[1] + (box[3] - box[1]) / 2])
        
        # Loop over the other indices of the remaining boxes
        for j in indices:
            if i != j: # Avoid comparing with itself
                box_b = boxes[j]
                # Get the center point of the other box
                center_b = np.array([box_b[0] + (box_b[2] - box_b[0]) / 2, box_b[1] + (box_b[3] - box_b[1]) / 2])
               
                # Compute the distance between the two points
                distance = euclidean_distance(center_a, center_b)
                
                # Check if the distance is below the threshold
                if distance < distance_threshold:
                    # Draw a red line between the two points
                    v+=1
                    cv2.line(frame, (int(center_a[0]), int(center_a[1])), (int(center_b[0]), int(center_b[1])), (0, 0, 255), 2)

    # Show the output frame
    cv2.imshow("Social Distance Detector", frame)
    cv2.putText(frame,'Number of Violations : '+str(v),(80,frame.shape[0]-10),cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,255),3)
    
    cv2.putText(frame,"FPS :"+ fps, (7,70), cv2.FONT_HERSHEY_SIMPLEX, 3, (200,0,0), 3, cv2.LINE_AA)
    writer.write(frame)
    
    
    
    
    # Wait for a key press to exit
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Release the video and destroy the windows
video.release()
cv2.destroyAllWindows()
