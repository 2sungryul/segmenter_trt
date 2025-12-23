#include "segmenter_trt/SegmenterTRT.hpp"
#include <iostream>
#include <chrono>

int main(int argc, char** argv) {
    /*if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <engine_file> <image_file>" << std::endl;
        return -1;
    }
    std::string enginePath = argv[1];
    std::string imagePath = argv[2];*/
    std::string enginePath = "/home/linux/ros2_ws/src/segmenter_trt/src/segmenter.engine";
    std::string imagePath = "/home/linux/ros2_ws/src/segmenter_trt/src/IMG_22.jpg";

    try {
        // Load image
        cv::Mat image = cv::imread(imagePath);
        if (image.empty()) {
            std::cerr << "Failed to load image: " << imagePath << std::endl;
            return -1;
        }

        std::cout << "Image loaded: " << image.cols << "x" << image.rows << std::endl;

        // Create inference engine
        SegmenterTRT segmenter(enginePath);

        // Run inference
        auto start = std::chrono::high_resolution_clock::now();
        cv::Mat segMap = segmenter.infer(image);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Inference time: " << duration.count() << " ms" << std::endl;

        // Visualize results
        cv::Mat coloredSegMap = segmenter.visualize(segMap);

        // Blend with original image
        cv::Mat blended;
        cv::addWeighted(image, 0.6, coloredSegMap, 0.4, 0, blended);

        // Save results
        cv::imwrite("segmentation_map.png", segMap);
        cv::imwrite("segmentation_colored.png", coloredSegMap);
        cv::imwrite("segmentation_blended.png", blended);

        std::cout << "Results saved:" << std::endl;
        std::cout << "  - segmentation_map.png (raw class IDs)" << std::endl;
        std::cout << "  - segmentation_colored.png (colored visualization)" << std::endl;
        std::cout << "  - segmentation_blended.png (blended with input)" << std::endl;

        // Display results (optional - comment out if running headless)
        cv::imshow("Original", image);
        cv::imshow("Segmentation", coloredSegMap);
        cv::imshow("Blended", blended);
        cv::waitKey(0);

    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}