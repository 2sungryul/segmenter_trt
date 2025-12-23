#ifndef SEGMENTER_TRT_H
#define SEGMENTER_TRT_H

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>


using namespace nvinfer1;

// Logger for TensorRT
class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override;
};

class SegmenterTRT {
private:
    std::unique_ptr<IRuntime> runtime;
    std::unique_ptr<ICudaEngine> engine;
    std::unique_ptr<IExecutionContext> context;

    void* buffers[2];  // input and output buffers
    cudaStream_t stream;

    int inputH, inputW, inputC;
    int outputH, outputW, numClasses;
    size_t inputSize, outputSize;

    std::string inputTensorName;
    std::string outputTensorName;

    // ImageNet normalization parameters
    static constexpr float mean[3] = { 0.485f, 0.456f, 0.406f };
    static constexpr float std[3] = { 0.229f, 0.224f, 0.225f };

public:
    SegmenterTRT(const std::string& enginePath);
    ~SegmenterTRT();

    // Disable copy constructor and assignment operator
    SegmenterTRT(const SegmenterTRT&) = delete;
    SegmenterTRT& operator=(const SegmenterTRT&) = delete;

    cv::Mat preprocess(const cv::Mat& img);
    cv::Mat infer(const cv::Mat& inputImage);
    cv::Mat visualize(const cv::Mat& segMap, int numColors = 150);
    cv::Mat argmaxGPU(float* deviceOutput);
    void WarmUp();
};

#endif // SEGMENTER_TRT_H