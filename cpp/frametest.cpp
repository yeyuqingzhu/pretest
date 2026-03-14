#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
cv::Mat kernel=cv::getStructuringElement(cv::MORPH_RECT,cv::Size(3,3));

//颜色筛选
cv::Mat FindColours(const cv::Mat& Input){
    cv::Mat HSVInput,Output,Mask1,Mask2,Mask,OpenMat,CloseMat;

    cv::Scalar OrangeLower1(175,100,100);
    cv::Scalar OrangeUpper1(180,255,255);
    cv::Scalar OrangeLower2(1,90,90);
    cv::Scalar OrangeUpper2(7,255,255);

    cv::cvtColor(Input,HSVInput,cv::COLOR_BGR2HSV);
    cv::inRange(HSVInput,OrangeLower1,OrangeUpper1,Mask1);
    cv::inRange(HSVInput,OrangeLower2,OrangeUpper2,Mask2);
    cv::bitwise_or(Mask1,Mask2,Mask);

    cv::morphologyEx(Mask,OpenMat,cv::MORPH_OPEN,kernel);
    cv::morphologyEx(OpenMat,Output,cv::MORPH_CLOSE,kernel);

    return Output;
}
int main(){
    cv::Mat framepart,OrangeMat;
    framepart=cv::imread("../视频文件/framepart.png");
    if(framepart.empty()){
        std::cout<<"未能成功加载";
        return -1;
    }else{
        std::cout<<"成功加载";
    }
    OrangeMat=FindColours(framepart);
    cv::imshow("TestWindow",OrangeMat);
    cv::waitKey(0);
    return 0;
}