#include<opencv2/opencv.hpp>
#include<iostream>

int main() {
    std::cout << "OpenCV 版本: " << CV_VERSION << std::endl;
    std::cout << cv::getBuildInformation() << std::endl;
    return 0;
}