#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <chrono>
#include <vector>
#include <omp.h>

template<typename T>
T clamp(T value, T low, T high) {
    return (value < low) ? low : ((value > high) ? high : value);
}

double cubic_kernel(double x) {
    x = std::abs(x);
    if (x < 1.0) {
        return 1.5 * x * x * x - 2.5 * x * x + 1.0;
    } else if (x < 2.0) {
        return -0.5 * x * x * x + 2.5 * x * x - 4.0 * x + 2.0;
    }
    return 0.0;
}

void custom_resize_cubic(const cv::Mat& src, cv::Mat& dst, cv::Size dsize) {
    dst.create(dsize, src.type());
    
    const int src_width = src.cols;
    const int src_height = src.rows;
    const int dst_width = dsize.width;
    const int dst_height = dsize.height;
    const int channels = src.channels();

    const double scale_x = static_cast<double>(src_width) / dst_width;
    const double scale_y = static_cast<double>(src_height) / dst_height;

    std::vector<double> coefficient_x(dst_width * 4);
    std::vector<int> index_x(dst_width * 4);

    // Precompute x-direction coefficients and indices
    for (int x = 0; x < dst_width; ++x) {
        double src_x = (x + 0.5) * scale_x - 0.5;
        int ix = static_cast<int>(src_x);
        double dx = src_x - ix;

        for (int n = -1; n <= 2; ++n) {
            int pixel_x = clamp(ix + n, 0, src_width - 1);
            coefficient_x[x * 4 + (n + 1)] = cubic_kernel(n - dx);
            index_x[x * 4 + (n + 1)] = pixel_x;
        }
    }

    // Resize
    for (int y = 0; y < dst_height; ++y) {
        double src_y = (y + 0.5) * scale_y - 0.5;
        int iy = static_cast<int>(src_y);
        double dy = src_y - iy;

        double coefficient_y[4];
        for (int n = -1; n <= 2; ++n) {
            coefficient_y[n + 1] = cubic_kernel(n - dy);
        }

        for (int x = 0; x < dst_width; ++x) {
            double sum[4] = {0};  // Assuming maximum 4 channels (e.g., RGBA)

            for (int ny = -1; ny <= 2; ++ny) {
                int pixel_y = clamp(iy + ny, 0, src_height - 1);
                double ky = coefficient_y[ny + 1];

                for (int nx = -1; nx <= 2; ++nx) {
                    int pixel_x = index_x[x * 4 + (nx + 1)];
                    double kx = coefficient_x[x * 4 + (nx + 1)];
                    double k = ky * kx;

                    for (int c = 0; c < channels; ++c) {
                        sum[c] += k * src.at<uchar>(pixel_y, pixel_x * channels + c);
                    }
                }
            }

            for (int c = 0; c < channels; ++c) {
                dst.at<uchar>(y, x * channels + c) = cv::saturate_cast<uchar>(sum[c]);
            }
        }
    }
}
void resizeImageNearest(const cv::Mat& inputImage, cv::Mat& outputImage, double xFactor, double yFactor) {
    int inputHeight = inputImage.rows;
    int inputWidth = inputImage.cols;
    int outputHeight = static_cast<int>(inputHeight * yFactor);
    int outputWidth = static_cast<int>(inputWidth * xFactor);

    outputImage = cv::Mat(outputHeight, outputWidth, inputImage.type());

    int xScale = static_cast<int>(xFactor * (1 << 16));
    int yScale = static_cast<int>(yFactor * (1 << 16));

    cv::parallel_for_(cv::Range(0, outputHeight), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
            int srcY = ((y << 16) + 0x8000) / yScale;

            for (int x = 0; x < outputWidth; ++x) {
                int srcX = ((x << 16) + 0x8000) / xScale;

                // Nearest neighbor interpolation
                int nearestX = srcX >> 16;
                int nearestY = srcY >> 16;

                // Clamp the coordinates to the input image boundaries
                nearestX = std::max(0, std::min(nearestX, inputWidth - 1));
                nearestY = std::max(0, std::min(nearestY, inputHeight - 1));

                outputImage.at<cv::Vec3b>(y, x) = inputImage.at<cv::Vec3b>(nearestY, nearestX);
            }
        }
    });
}
void custom_resize_linear_optimized(const cv::Mat& src, cv::Mat& dst, cv::Size dsize)
{
    dst.create(dsize, src.type());
    
    const int src_width = src.cols;
    const int src_height = src.rows;
    const int dst_width = dsize.width;
    const int dst_height = dsize.height;
    const int channels = src.channels();
    
    const double scale_x = static_cast<double>(src_width) / dst_width;
    const double scale_y = static_cast<double>(src_height) / dst_height;

    const unsigned char* src_data = src.data;
    unsigned char* dst_data = dst.data;
    const int src_step = static_cast<int>(src.step);
    const int dst_step = static_cast<int>(dst.step);

    #pragma omp parallel for
    for (int y = 0; y < dst_height; y++)
    {
        const int src_y = static_cast<int>(y * scale_y);
        const double dy = y * scale_y - src_y;
        const int src_y2 = (src_y + 1 < src_height) ? (src_y + 1) : src_y;

        unsigned char* dst_row = dst_data + y * dst_step;
        const unsigned char* src_row1 = src_data + src_y * src_step;
        const unsigned char* src_row2 = src_data + src_y2 * src_step;

        for (int x = 0; x < dst_width; x++)
        {
            const int src_x = static_cast<int>(x * scale_x);
            const double dx = x * scale_x - src_x;
            const int src_x2 = (src_x + 1 < src_width) ? (src_x + 1) : src_x;

            const double weight_tl = (1.0 - dx) * (1.0 - dy);
            const double weight_tr = dx * (1.0 - dy);
            const double weight_bl = (1.0 - dx) * dy;
            const double weight_br = dx * dy;

            for (int c = 0; c < channels; c++)
            {
                const int index = x * channels + c;
                const unsigned char tl = src_row1[src_x * channels + c];
                const unsigned char tr = src_row1[src_x2 * channels + c];
                const unsigned char bl = src_row2[src_x * channels + c];
                const unsigned char br = src_row2[src_x2 * channels + c];

                double val = weight_tl * tl + weight_tr * tr + 
                             weight_bl * bl + weight_br * br;

                dst_row[index] = cv::saturate_cast<uchar>(val);
            }
        }
    }
}


    int main() {
    cv::Mat image = cv::imread("G178_2 -1080.BMP");
    if (image.empty()) {
        std::cout << "Could not read the image" << std::endl;
        return 1;
    }
    std::cout << "Original Height and Width: " << image.rows << "x" << image.cols << std::endl;

    cv::Size dsize(image.cols / 2, image.rows / 2);
    cv::Mat resized_nearest, resized_linear, resized_cubic;
    cv::Mat custom_nearest, custom_linear, custom_cubic;

    // OpenCV inbuilt functions
    cv::resize(image, resized_nearest, dsize, 0, 0, cv::INTER_NEAREST);
    cv::resize(image, resized_linear, dsize, 0, 0, cv::INTER_LINEAR);
    cv::resize(image, resized_cubic, dsize, 0, 0, cv::INTER_CUBIC);

    // Save OpenCV resized images
    cv::imwrite("opencv_nearest.bmp", resized_nearest);
    cv::imwrite("opencv_linear.bmp", resized_linear);
    cv::imwrite("opencv_cubic.bmp", resized_cubic);

    // Measure OpenCV performance (1000 iterations)
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i)
        cv::resize(image, resized_nearest, dsize, 0, 0, cv::INTER_NEAREST);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time taken for 1000 iterations using cv::resize (INTER_NEAREST): " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i)
        cv::resize(image, resized_linear, dsize, 0, 0, cv::INTER_LINEAR);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Time taken for 1000 iterations using cv::resize (INTER_LINEAR): " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i)
        cv::resize(image, resized_cubic, dsize, 0, 0, cv::INTER_CUBIC);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Time taken for 1000 iterations using cv::resize (INTER_CUBIC): " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

    // Custom functions
    resizeImageNearest(image, custom_nearest, 0.5, 0.5);
    custom_resize_linear_optimized(image, custom_linear, dsize);
    custom_resize_cubic(image, custom_cubic, dsize);

    // Save custom resized images
    cv::imwrite("custom_nearest.bmp", custom_nearest);
    cv::imwrite("custom_linear.bmp", custom_linear);
    cv::imwrite("custom_cubic.bmp", custom_cubic);

    // Measure custom functions performance (1000 iterations)
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i)
        resizeImageNearest(image, custom_nearest, 0.5, 0.5);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Time taken for 1000 iterations using custom nearest neighbor: " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i)
        custom_resize_linear_optimized(image, custom_linear, dsize);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Time taken for 1000 iterations using custom linear interpolation: " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i)
        custom_resize_cubic(image, custom_cubic, dsize);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Time taken for 1000 iterations using custom cubic interpolation: " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

    return 0;
}
