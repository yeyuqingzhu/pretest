#include <opencv2/opencv.hpp>
#include <iostream>

cv::Mat frame, hsv, mask;
int low_H = 5, low_S = 100, low_V = 100;
int high_H = 15, high_S = 255, high_V = 255;

void onTrackbar(int, void*) {
    cv::Scalar lower(low_H, low_S, low_V);
    cv::Scalar upper(high_H, high_S, high_V);
    cv::inRange(hsv, lower, upper, mask);

    cv::Mat res;
    cv::bitwise_and(frame, frame, res, mask); // 在原图上显示颜色区域

    cv::imshow("掩码", mask);
    cv::imshow("结果", res);
}

int main() {
    frame = cv::imread("../视频文件/framepart.png");
    if (frame.empty()) {
        std::cerr << "无法加载图像" << std::endl;
        return -1;
    }

    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

    cv::namedWindow("调参");
    cv::createTrackbar("低H", "调参", &low_H, 180, onTrackbar);
    cv::createTrackbar("高H", "调参", &high_H, 180, onTrackbar);
    cv::createTrackbar("低S", "调参", &low_S, 255, onTrackbar);
    cv::createTrackbar("高S", "调参", &high_S, 255, onTrackbar);
    cv::createTrackbar("低V", "调参", &low_V, 255, onTrackbar);
    cv::createTrackbar("高V", "调参", &high_V, 255, onTrackbar);

    onTrackbar(0, nullptr);  // 显示初始效果

    cv::imshow("原始图像", frame);
    cv::waitKey(0);

    return 0;
}