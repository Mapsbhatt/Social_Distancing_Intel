# Model Performance Comparison: OpenVINO vs. Without OpenVINO

This repository contains code and instructions to compare the performance of a model using OpenVINO and without OpenVINO. The goal is to analyze the impact of OpenVINO on the model's performance in terms of speed and resource utilization.

## Table of Contents
- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Performance Metrics](#performance-metrics)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

In this project, we aim to evaluate the performance of a model when using OpenVINO, a toolkit for optimizing deep learning models for deployment on various hardware architectures. OpenVINO provides optimizations such as model quantization, model compression, and hardware-specific optimizations, which can potentially enhance the inference speed and resource utilization of the model.

## Prerequisites

To run the code and reproduce the performance comparison, the following prerequisites are required:

- Python (version 3.8.2 and above)
- OpenVINO toolkit (version 2022.3.1)
- Model weights and configurations (provide details on where to obtain or download the model)

## Installation

1. Clone this repository:

   ``` shell
   git clone [https://github.com/your-username/your-repository.git](https://github.com/Mapsbhatt/Manan_Bhatt-MIT-Social_Distancing-intelunnati.git)
   ```

2. Install the required dependencies:
     ```python
   pip install -r requirements.txt
   ```

3. Configure OpenVINO:
   Follow the official OpenVINO documentation to install and configure OpenVINO for your specific hardware and operating system.
   
4. Download the model weights and configurations:
   Download the necessary model files and place them in the appropriate directory.


## Usage

Without OpenVINO:
Run the model without OpenVINO optimization and note the performance metrics.

With OpenVINO:
Integrate the model with OpenVINO optimizations and compare the performance metrics with the previous run.

The basic steps for opimisation and conversion are:
1. Model Selection: Choose the model you want to optimize and ensure it is compatible with Intel OpenVINO.
2. Model Optimization: Optimize the model using techniques like quantization and precision adjustment to improve performance and resource utilization.
3. Model Conversion: Use the OpenVINO Model Optimizer tool to convert the optimized model to the Intermediate Representation (IR) format.
4. Choose Target Device: Determine the target hardware device for inference, such as Intel CPUs, GPUs, or FPGAs.
Inference Code Integration: Integrate the converted model into your application using the OpenVINO Inference Engine APIs.

## Model Specifications

The model used for the performance comparison is YOLOv5s, a popular object detection model. YOLOv5 is a state-of-the-art real-time object detection model that achieves high accuracy while maintaining fast inference speeds.

Model: YOLOv5s
Architecture: YOLO (You Only Look Once)
Backbone: CSPDarknet53
Input Size: 416x416 pixels (configurable)
Output: Bounding boxes and class probabilities
The YOLOv5s model is chosen for its balance between accuracy and speed, making it suitable for real-time applications. However, please note that you can adapt the instructions and code provided in this repository to compare the performance of other models with and without OpenVINO.

For more details on the YOLOv5 model and its architecture, refer to the official YOLOv5 repository: https://github.com/ultralytics/yolov5

## Analyze the performance
Compare the performance metrics displayed after running the model with and without OpenVINO. Evaluate factors such as inference time, CPU/GPU utilization, and memory usage.

These performance metrics provide a comprehensive view of the model's performance when using OpenVINO compared to without OpenVINO. Monitoring and comparing these metrics help assess the impact of OpenVINO optimizations on speed, resource utilization, memory efficiency, scalability, and energy consumption.

1. Inference Time: OpenVINO optimizations can potentially improve the model's inference speed by 1.5x to 4x compared to running the model without OpenVINO.
2. CPU/GPU Utilization: OpenVINO optimizations can lead to higher CPU/GPU utilization, often ranging from 60% to 90%, indicating improved resource utilization.
3. Memory Usage: OpenVINO optimizations can sometimes reduce memory usage by up to 50% compared to running the model without OpenVINO. However, memory usage can vary depending on the model's architecture and the specific optimizations applied.
4. Throughput: With OpenVINO optimizations, the model's throughput can increase by approximately 2x to 5x compared to running the model without OpenVINO. This allows for processing a higher volume of data in a given timeframe.


## License
This project is licensed under the MIT License.

## Contact
For more information, please contact Manan Bhatt at mapsbhatt.mb@gmail.com.
   

