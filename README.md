# Assignment-for-Deep-Edge
Assignment submission for DeepEdge internship in Computer Vision

## Introduction
This project demonstrates image resizing using OpenCV and custom implementations of interpolation methods.

## Dependencies
- OpenCV (>=4.x)
- C++ compiler with C++11 support
- CMake (for building the project)

## Building the Project
1. Ensure OpenCV is installed on your system.
2. Navigate to the project directory.
3. Create a build directory and navigate into it:
   ```sh
   mkdir build
   cd build

## Observed Outputs
The following timings were observed during the execution of the program:

OpenCV Functions
Time taken for 1000 iterations using INTER_NEAREST: 349 ms
Time taken for 1000 iterations using INTER_LINEAR: 498 ms
Time taken for 1000 iterations using INTER_CUBIC: 3430 ms
Custom Functions
Time taken for 1000 iterations using Custom Nearest Neighbor: 1872 ms
Time taken for 1000 iterations using Custom Linear Interpolation: 37349 ms
Time taken for 1000 iterations using Custom Cubic Interpolation: [ ] ms


Output Files
opencv_nearest.bmp
opencv_linear.bmp
opencv_cubic.bmp
custom_nearest.bmp
custom_linear.bmp
custom_cubic.bmp
These files will be generated in the project directory after running the application.
