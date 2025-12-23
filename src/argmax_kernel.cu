// argmax_kernel.cu
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUDA kernel for argmax operation
// Input: [numClasses, height, width] in NCHW format
// Output: [height, width] with class indices
__global__ void argmaxKernel(
    const float* input,
    unsigned char* output,
    int numClasses,
    int height,
    int width
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int pixelIdx = y * width + x;

    // Find max class for this pixel
    float maxVal = input[pixelIdx]; // Class 0
    int maxClass = 0;

    for (int c = 1; c < numClasses; c++) {
        int idx = c * height * width + pixelIdx;
        float val = input[idx];
        if (val > maxVal) {
            maxVal = val;
            maxClass = c;
        }
    }

    output[pixelIdx] = static_cast<unsigned char>(maxClass);
}

// Host function to launch kernel
extern "C"
void launchArgmaxKernel(
    const float* d_input,
    unsigned char* d_output,
    int numClasses,
    int height,
    int width,
    cudaStream_t stream
) {
    // Configure grid and block dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y
    );

    // Launch kernel
    argmaxKernel << <gridSize, blockSize, 0, stream >> > (
        d_input, d_output, numClasses, height, width
        );
}