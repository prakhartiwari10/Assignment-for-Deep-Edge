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
2. Clone the repository or unzip the compressed file into a directory.
3. Navigate to the project directory.
4. Create a build directory and navigate into it:
   ```sh
   mkdir build
   cd build
5. Run CMake to configure the project:
   ```sh
   cmake ..
7. Build the project:
   ```sh
   make
   
## Running the Application

   Ensure the input image G178_2-1080.BMP is in the project directory.
   Run the executable:
   ```sh
   ./AssignmentCombined
   ```

## Observed Outputs
The following timings were observed during the execution of the program:

OpenCV Functions
Time taken for 1000 iterations using INTER_NEAREST: 349 ms
Time taken for 1000 iterations using INTER_LINEAR: 498 ms
Time taken for 1000 iterations using INTER_CUBIC: 3430 ms
Custom Functions
Time taken for 1000 iterations using Custom Nearest Neighbor: 1872 ms
Time taken for 1000 iterations using Custom Linear Interpolation: 37349 ms
Time taken for 1000 iterations using Custom Cubic Interpolation: 145196 ms


Output Files
opencv_nearest.bmp
opencv_linear.bmp
opencv_cubic.bmp
custom_nearest.bmp
custom_linear.bmp
custom_cubic.bmp
These files will be generated in the project directory after running the application.

## Areas that can still be improved:
I couldn't come up with an optimized method for cubic interpolation, so that can be improved. Also, one thing we can do is utilize the special case, i.e, if we already know the resizing factor, we can make function specifically for that case which will crop the image into width/2 x height/2, but that will be just specific to the case and won't generalize to other cases.
