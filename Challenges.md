# Social Distancing using OpenCV and Intel OpenVINO Optimisation

## Difficulties Faced

This README file documents the various challenges and difficulties encountered during the development of the project. It provides insights into the problems faced and the solutions or workarounds implemented to overcome them.

### 1. Lack of Integration with Different System Architectures
Throughout the project, we encountered difficulties with certain plugins of the Intel OpenVINO toolkit, particularly when using systems that were not equipped with Intel chipsets. Unfortunately, both of the laptops available for development lacked Intel chipsets, making it challenging to ensure smooth functioning of these plugins. Although the project offered access to the Intel DevCloud, the overall process of troubleshooting and resolving compatibility issues proved to be cumbersome.

### 2. DevCloud Complexity:
1. While the Intel DevCloud provided an alternative environment for development and testing, navigating its setup and configuration added complexity to the overall workflow. The process of adapting the project to run on the DevCloud infrastructure and ensuring compatibility with non-Intel systems proved time-consuming and resource-intensive.
Integrating multiple components or services can be complex, and compatibility issues can arise. During the project, compatibility issues were encountered while integrating different modules or third-party services. Extensive debugging and testing were performed to identify and resolve these issues. Regular communication with the relevant communities or support forums also helped in finding solutions or alternative approaches.

2. During the project, we encountered difficulties with the availability of certain prerequisites and dependencies on the DevCloud notebook environment. Despite attempting manual installations, we consistently encountered errors, necessitating the exploration of alternative methods to achieve the desired results.

### 3. Performance Optimization

1. During the normal workflow of using the YOLOv5s model with PyTorch or ONNX for the social distancing application, the system performed well, delivering the expected results and achieving satisfactory frame rates. However, we encountered a significant challenge when attempting to export the correct output video after applying conversion and optimization techniques. Unfortunately, regardless of the codec used, the resulting output video consistently turned out to be corrupted.
2. Throughout the project, we faced several challenges that hindered the accuracy and performance comparison between two methods. One major obstacle was the inability to obtain the desired output from the converted format, preventing us from exploring the performance benchmarks offered by the OpenVINO toolkit. Despite these challenges, we remain determined to overcome them and unlock the intriguing potential of the toolkit. Ongoing efforts are focused on obtaining the correct output, enabling us to evaluate performance benchmarks and optimize the social distancing application.
As the project grew in complexity, performance became a significant concern. The application experienced slowdowns or bottlenecks in certain scenarios. Profiling tools and performance analysis were utilized to identify the critical areas causing performance issues. Strategies such as code refactoring, caching, and algorithmic improvements were implemented to optimize the performance.


In conclusion, the social distancing project has been an immensely beneficial undertaking, enriched by the utilization and interaction with the OpenVINO toolkit. Despite the challenges faced, the integration of the OpenVINO toolkit allowed us to explore optimization and acceleration techniques for our computer vision models. While we didn't achieve the expected output in terms of accuracy and performance, the toolkit provided valuable insights and opportunities for improvement. It served as a catalyst for exploring different strategies and understanding the intricacies of model optimization. Moreover, the constant weekly support and guidance from our mentors played a pivotal role in overcoming challenges and achieving significant progress. Their expertise and assistance helped us navigate complexities, troubleshoot issues, and further explore the potential of the project. We are grateful for their unwavering commitment and support, which greatly contributed to the success of our endeavor. Together with the OpenVINO toolkit, their guidance has made this project an invaluable learning experience


