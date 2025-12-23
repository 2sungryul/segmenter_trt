cmake_minimum_required(VERSION 3.18)
project(segmenter_trt LANGUAGES CXX CUDA)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CUDA_STANDARD 17)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find ROS2 dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)

# Find other dependencies
#find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

# TensorRT paths
set(TensorRT_INCLUDE_DIRS "/usr/include/x86_64-linux-gnu")
set(TensorRT_LIBRARIES 
    "/usr/lib/x86_64-linux-gnu/libnvinfer.so"
    "/usr/lib/x86_64-linux-gnu/libnvonnxparser.so"
)
set(CUDA_INCLUDE_DIRS "/usr/local/cuda-12.9/include")
#link_directories(/usr/local/cuda-12.9/lib64)

# Include directories
include_directories(
    include
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIRS}
)

# ROS2 Node - inference node using single image
add_executable(segmenter_node src/SegmenterNode.cpp src/SegmenterTRT.cpp src/argmax_kernel.cu)
ament_target_dependencies(segmenter_node rclcpp sensor_msgs cv_bridge)
target_link_libraries(segmenter_node ${OpenCV_LIBS} ${TensorRT_LIBRARIES} cudart)

# Set CUDA architectures
set_property(TARGET segmenter_node PROPERTY CUDA_ARCHITECTURES 75 80 86 89)

# ROS2 image publisher
add_executable(pub src/pub.cpp)
ament_target_dependencies(pub rclcpp sensor_msgs cv_bridge)
target_link_libraries(pub ${OpenCV_LIBS})

# ROS2 image subscription and model inference
add_executable(sub src/sub.cpp src/SegmenterTRT.cpp src/argmax_kernel.cu)
ament_target_dependencies(sub rclcpp sensor_msgs)
target_link_libraries(sub ${OpenCV_LIBS} ${TensorRT_LIBRARIES} cudart)

# Set CUDA architectures
set_property(TARGET sub PROPERTY CUDA_ARCHITECTURES 75 80 86 89)

# Install targets
install(TARGETS
    segmenter_node
    pub
    sub
    DESTINATION lib/${PROJECT_NAME}
)

ament_package()
