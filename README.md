# segmenter_trt

segmentation model : seg_b_mask_16_ade20k

dependencies : wsl2-ubuntu24.04, ROS2 Jazzy, TensorRT 10.14.1, Cuda 12.9.1, cudnn 8.9.7

TensorRT engine file : https://drive.google.com/drive/folders/1CQwC1yTIimgzU9wkLwqpfBiQK4upOc9q

segmenter_node : inference node using single image file

pub : publish a topic with image captured from video file(mp4)

sub : inference node using image topic received from pub node

engine inference time < about 15msec on RTX 4070 Ti super

total callback processing time < about 25msec on RTX 4070 Ti super
