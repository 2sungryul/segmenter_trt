#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "opencv2/opencv.hpp"
#include <memory>
#include <functional>
#include <iostream>
#include "segmenter_trt/SegmenterTRT.hpp"
#include <chrono>
using std::placeholders::_1;

std::string enginePath = "/home/linux/ros2_ws/src/segmenter_trt/src/segmenter.engine";
std::string imagePath = "/home/linux/ros2_ws/src/segmenter_trt/src/IMG_22.jpg";

class Sub : public rclcpp::Node
{
private:
    rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr sub_;
    std::shared_ptr<SegmenterTRT> segmenter_;
    void sub_callback(const sensor_msgs::msg::CompressedImage::SharedPtr msg) const;
public:
    Sub();
};

Sub::Sub() : Node("imagesub")
{
    auto qos_profile = rclcpp::QoS(rclcpp::KeepLast(10));
    sub_ = this->create_subscription<sensor_msgs::msg::CompressedImage>("image/compressed", qos_profile, std::bind(&Sub::sub_callback, this, _1));
    segmenter_ = std::make_shared<SegmenterTRT>(enginePath);
}

void Sub::sub_callback(const sensor_msgs::msg::CompressedImage::SharedPtr msg) const
{
    cv::Mat frame = cv::imdecode(cv::Mat(msg->data), cv::IMREAD_COLOR);
    RCLCPP_INFO(this->get_logger(), "Received Image : %s,%d,%d", msg->format.c_str(),frame.rows,frame.cols);
    
    // Run inference
    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat segMap = segmenter_->infer(frame);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Inference time: " << duration.count() << " ms" << std::endl;

    // Visualize results
    cv::Mat coloredSegMap = segmenter_->visualize(segMap);

    // Blend with original image
    cv::Mat blended;
    cv::addWeighted(frame, 0.6, coloredSegMap, 0.4, 0, blended);

    // Display results (optional - comment out if running headless)
    cv::imshow("Original", frame);
    cv::imshow("Segmentation", coloredSegMap);
    cv::imshow("Blended", blended);    
    cv::waitKey(1);
    
    auto end2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start);
    std::cout << "callback processing time: " << duration2.count() << " ms" << std::endl;
}

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<Sub>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
