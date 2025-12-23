#include "segmenter_trt/SegmenterTRT.hpp"
#include <NvOnnxParser.h>
#include <fstream>
#include <iostream>
#include <algorithm>

// External CUDA function
extern "C" void launchArgmaxKernel(
    const float* d_input,
    unsigned char* d_output,
    int numClasses,
    int height,
    int width,
    cudaStream_t stream
);

// Logger implementation
void Logger::log(Severity severity, const char* msg) noexcept {
    if (severity <= Severity::kWARNING)
        std::cout << msg << std::endl;
}

// SegmenterTRT constructor
SegmenterTRT::SegmenterTRT(const std::string& enginePath) {
    // Load engine from file
    std::ifstream file(enginePath, std::ios::binary);
    if (!file.good()) {
        throw std::runtime_error("Failed to open engine file: " + enginePath);
    }

    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);

    std::vector<char> engineData(size);
    file.read(engineData.data(), size);
    file.close();

    // Create runtime and deserialize engine
    static Logger gLogger;
    runtime.reset(createInferRuntime(gLogger));
    engine.reset(runtime->deserializeCudaEngine(engineData.data(), size));
    context.reset(engine->createExecutionContext());

    // Get tensor names
    int numIOTensors = engine->getNbIOTensors();
    for (int i = 0; i < numIOTensors; i++) {
        const char* tensorName = engine->getIOTensorName(i);
        TensorIOMode mode = engine->getTensorIOMode(tensorName);

        if (mode == TensorIOMode::kINPUT) {
            inputTensorName = tensorName;
        }
        else if (mode == TensorIOMode::kOUTPUT) {
            outputTensorName = tensorName;
        }
    }

    std::cout << "Input tensor name: " << inputTensorName << std::endl;
    std::cout << "Output tensor name: " << outputTensorName << std::endl;

    // Get input/output dimensions
    auto inputDims = engine->getTensorShape(inputTensorName.c_str());
    auto outputDims = engine->getTensorShape(outputTensorName.c_str());

    inputC = inputDims.d[1];
    inputH = inputDims.d[2];
    inputW = inputDims.d[3];

    numClasses = outputDims.d[1];
    outputH = outputDims.d[2];
    outputW = outputDims.d[3];

    inputSize = 1 * inputC * inputH * inputW * sizeof(float);
    outputSize = 1 * numClasses * outputH * outputW * sizeof(float);

    // Allocate GPU memory
    cudaMalloc(&buffers[0], inputSize);
    cudaMalloc(&buffers[1], outputSize);
    cudaStreamCreate(&stream);

    std::cout << "Engine loaded successfully!" << std::endl;
    std::cout << "Input shape: [1, " << inputC << ", " << inputH << ", " << inputW << "]" << std::endl;
    std::cout << "Output shape: [1, " << numClasses << ", " << outputH << ", " << outputW << "]" << std::endl;

    WarmUp();

}

// SegmenterTRT destructor
SegmenterTRT::~SegmenterTRT() {
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);
    cudaStreamDestroy(stream);
}

void SegmenterTRT::WarmUp() {
    
    std::cout << "Warming up TensorRT engine..." << std::endl;

    // 더미 입력 데이터 생성
    std::vector<float> dummy_input(1 * inputC * inputW * inputH, 0.5f);

    // Set input tensor address
    context->setTensorAddress(inputTensorName.c_str(), buffers[0]);
    // Set output tensor address
    context->setTensorAddress(outputTensorName.c_str(), buffers[1]);
    
    // 첫 실행들 (느림)
    cudaMemcpyAsync(buffers[0], dummy_input.data(), inputSize,cudaMemcpyHostToDevice, stream);

    bool success = context->enqueueV3(stream);
    if (!success) {
        std::cerr << "Error: Warmup Inference failed" << std::endl;
        return;
    }

    cudaStreamSynchronize(stream);    
    
    std::cout << "Warmup complete!" << std::endl;
}

// Preprocess image
cv::Mat SegmenterTRT::preprocess(const cv::Mat& img) {
    cv::Mat resized, rgb, normalized;

    // Resize to model input size
    cv::resize(img, resized, cv::Size(inputW, inputH));

    // Convert BGR to RGB
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    // Convert to float and normalize to [0, 1]
    rgb.convertTo(normalized, CV_32FC3, 1.0 / 255.0);

    // Apply ImageNet normalization
    std::vector<cv::Mat> channels(3);
    cv::split(normalized, channels);

    for (int c = 0; c < 3; c++) {
        channels[c] = (channels[c] - mean[c]) / std[c];
    }

    cv::Mat merged;
    cv::merge(channels, merged);

    return merged;
}

// Run inference
cv::Mat SegmenterTRT::infer(const cv::Mat& inputImage) {
    auto start = std::chrono::high_resolution_clock::now();
    
    // Preprocess
    cv::Mat preprocessed = preprocess(inputImage);
    
    //auto start = std::chrono::high_resolution_clock::now();
    
    // Convert to NCHW format
    std::vector<float> inputData(inputC * inputH * inputW);
    std::vector<cv::Mat> channels(3);
    cv::split(preprocessed, channels);

    for (int c = 0; c < inputC; c++) {
        memcpy(inputData.data() + c * inputH * inputW,
            channels[c].data,
            inputH * inputW * sizeof(float));
    }

    // Copy input to GPU
    cudaMemcpyAsync(buffers[0], inputData.data(), inputSize,
        cudaMemcpyHostToDevice, stream);

    // Run inference using enqueueV3
    context->setTensorAddress(inputTensorName.c_str(), buffers[0]);
    context->setTensorAddress(outputTensorName.c_str(), buffers[1]);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Preprocessing time: " << duration.count() << " ms" << std::endl;

    auto start1 = std::chrono::high_resolution_clock::now();
    bool success = context->enqueueV3(stream);
    if (!success) {
        std::cerr << "Error: Inference failed" << std::endl;
        return {};
    }
    cudaStreamSynchronize(stream);
    auto end1 = std::chrono::high_resolution_clock::now();
    
    auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1);
    std::cout << "Inference(enqueueV3) time: " << duration1.count() << " ms" << std::endl;    

    auto start2 = std::chrono::high_resolution_clock::now();

    // Post-process: Get class with maximum probability for each pixel
    // Use GPU-based argmax for better performance
    cv::Mat segMap = argmaxGPU(static_cast<float*>(buffers[1]));
    
    //auto end2 = std::chrono::high_resolution_clock::now();
    
    // Resize back to original image size
    cv::Mat finalSegMap;
    cv::resize(segMap, finalSegMap, inputImage.size(), 0, 0, cv::INTER_NEAREST);
    
    auto end2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2);
    std::cout << "Postprocessing time: " << duration2.count() << " ms" << std::endl;

    return finalSegMap;
}

// Visualize segmentation map
// https://github.com/rstrudel/segmenter/blob/master/segm/data/config/ade20k.yml
cv::Mat SegmenterTRT::visualize(const cv::Mat& segMap, int numColors) {
    cv::Mat colorMap(segMap.size(), CV_8UC3);

    // Generate random colors for each class
    std::vector<cv::Vec3b> colors(numColors);
    cv::RNG rng(12345);
    for (int i = 0; i < numColors; i++) {
        colors[i] = cv::Vec3b(rng.uniform(0, 256),
            rng.uniform(0, 256),
            rng.uniform(0, 256));
    }
    colors[0] = cv::Vec3b(0, 0, 0); // Background is black

    for (int h = 0; h < segMap.rows; h++) {
        for (int w = 0; w < segMap.cols; w++) {
            int classId = segMap.at<uchar>(h, w);
            colorMap.at<cv::Vec3b>(h, w) = colors[classId % numColors];
        }
    }

    return colorMap;
}

// GPU-based argmax (fastest method)
cv::Mat SegmenterTRT::argmaxGPU(float* deviceOutput) {
    // Allocate GPU memory for output segmentation map
    unsigned char* d_segMap;
    cudaMalloc((void**) & d_segMap, outputH * outputW * sizeof(unsigned char));

    // Launch CUDA kernel for argmax
    launchArgmaxKernel(deviceOutput, d_segMap, numClasses, outputH, outputW, stream);

    // Copy result back to CPU
    cv::Mat segMap(outputH, outputW, CV_8UC1);
    cudaMemcpyAsync(segMap.data, d_segMap, outputH * outputW, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaFree(d_segMap);

    return segMap;
}