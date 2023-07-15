import cv2
import numpy as np
import onnxruntime

# Load the pretrained person detection model
person_detection_model = onnxruntime.InferenceSession('path_to_your_model.onnx')

# Define the safe distance threshold
safe_distance_threshold = 6.0  # Adjust according to your requirements

# Load and preprocess the input image/frame
image = cv2.imread('path_to_your_image.jpg')
# Perform any necessary preprocessing steps such as resizing, normalization, etc.

# Run the person detection model on the preprocessed image
input_name = person_detection_model.get_inputs()[0].name
output_name = person_detection_model.get_outputs()[0].name
input_data = np.expand_dims(image, axis=0)
output = person_detection_model.run([output_name], {input_name: input_data})

# Extract bounding box coordinates for detected people
boxes = output[0]  # Assuming the output is a list of bounding box coordinates

# Calculate pairwise distances between people
num_people = len(boxes)
distances = np.zeros((num_people, num_people))
for i in range(num_people):
    for j in range(i + 1, num_people):
        distance = calculate_distance(boxes[i], boxes[j])  # Implement your distance calculation method
        distances[i, j] = distance
        distances[j, i] = distance

# Identify pairs of people not maintaining safe distance
unsafe_pairs = np.argwhere(distances < safe_distance_threshold)

# Visualize the results
for i, box in enumerate(boxes):
    # Draw bounding box around each person
    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    
for pair in unsafe_pairs:
    # Draw lines connecting unsafe pairs of people
    cv2.line(image, (boxes[pair[0], 0], boxes[pair[0], 1]), (boxes[pair[1], 0], boxes[pair[1], 1]), (0, 0, 255), 2)

# Display the final image with visualizations
cv2.imshow('Social Distancing Model', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
