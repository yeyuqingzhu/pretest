#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
cv::Mat kernel=cv::getStructuringElement(cv::MORPH_RECT,cv::Size(3,3));

//颜色筛选
cv::Mat FindColours(const cv::Mat& Input){
    cv::Mat HSVInput,Output,Mask,CloseMat;

    cv::Scalar OrangeLower(5,100,120);
    cv::Scalar OrangeUpper(25,255,255);

    cv::cvtColor(Input,HSVInput,cv::COLOR_BGR2HSV);
    cv::inRange(HSVInput,OrangeLower,OrangeUpper,Mask);

    cv::morphologyEx(Mask,CloseMat,cv::MORPH_CLOSE,kernel);
    cv::morphologyEx(CloseMat,Output,cv::MORPH_OPEN,kernel);

    return Output;
}

//连通域提取
int FindContours(const cv::Mat& Input,cv::Mat& Component,cv::Mat& results,cv::Mat& centers){      //输入：待处理图像（灰度）；输出：连通区域，参数，质心
    int num;
    num=cv::connectedComponentsWithStats(Input,Component,results,centers,8);

    return num;
}

//判断并输出圆
bool BoolCircle(int num,cv::Mat& Input,cv::Mat& results,cv::Mat& centers,cv::Point2f& point,double& radius){
    int result=-1;
    double best=0;
    bool found=false;

    for(int i=1;i<num;i++){
        if(results.at<int>(i,4)<200)continue;

        cv::Mat Component=(Input==i);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(Component,contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);
        if(contours.empty())continue;
        double area=cv::contourArea(contours[0]);
        double perimeter=cv::arcLength(contours[0],true);
        double circularity=4*CV_PI*area/(perimeter*perimeter);

        if(circularity>0.8&&circularity>best){
            found=true;
            point.x=centers.at<double>(i,0);
            point.y=centers.at<double>(i,1);
            best=circularity;
            radius=std::sqrt(area/CV_PI);
        }
    }
    return found;
}

//图像矩
bool findcenter(const cv::Mat&Input,cv::Point2f& point,double& radius){
    cv::Mat Blur;
    cv::GaussianBlur(Input,Blur,cv::Size(11,11),0);
    cv::Moments m=cv::moments(Blur,true);
    if(m.m00<100)return false;
    point.x=m.m10/m.m00;
    point.y=m.m01/m.m00;
    radius=std::sqrt(m.m00/CV_PI);
    return true;
}

//卡尔曼滤波器
cv::Point2f KalmanPredict(cv::KalmanFilter& kf, cv::Point2f measurement, bool found){
    // 预测
    cv::Mat prediction=kf.predict();

    float pred_x=prediction.at<float>(0);
    float pred_y=prediction.at<float>(1);

    // 如果检测到了目标则进行校正
    if(found){
        cv::Mat meas(2,1,CV_32F);
        meas.at<float>(0)=measurement.x;
        meas.at<float>(1)=measurement.y;

        kf.correct(meas);
    }

    return cv::Point2f(pred_x, pred_y);
}

int main(){
    //获取文件
    cv::VideoCapture cap("../视频文件/video.mp4");
    if (!cap.isOpened()){
            std::cout<<"视频未成功加载"<<std::endl;
            return -1;
        }else{
            std::cout<<"视频成功加载"<<std::endl;
        }
    cv::Mat original,gray,OrangeMat,CannyMat,Component,results,centers;
    cv::Point2f measurement,predict;
    bool found=false,firstfound=false;
    int num=0;
    double radius;
    //-----------------卡尔曼滤波---------------------
        cv::KalmanFilter kf(4,2,0);

        // 状态转移矩阵
        kf.transitionMatrix=(cv::Mat_<float>(4,4)<<
        1,0,1,0,
        0,1,0,1,
        0,0,1,0,
        0,0,0,1);

        // 测量矩阵
        kf.measurementMatrix=cv::Mat::zeros(2,4,CV_32F);
        kf.measurementMatrix.at<float>(0,0) = 1;
        kf.measurementMatrix.at<float>(1,1) = 1;

        // 噪声矩阵
        setIdentity(kf.processNoiseCov,cv::Scalar::all(1e-2));
        setIdentity(kf.measurementNoiseCov,cv::Scalar::all(1e-2));
        setIdentity(kf.errorCovPost,cv::Scalar::all(1));


    //---------------------------------------------
    //帧处理
    while(true){
        //预处理
        cv::Mat frame;
        cap>>frame;
        if(frame.empty())break;

        OrangeMat=FindColours(frame);
        //num=FindContours(OrangeMat,Component,results,centers);
        //found=BoolCircle(num,Component,results,centers,measurement,radius);
        found=findcenter(OrangeMat,measurement,radius);
        
        predict=KalmanPredict(kf,measurement,found);

        std::cout<<found;

        //绘制并显示结果
        if(found){
            if(firstfound==false){
                //初始状态
                firstfound=true;
                kf.statePost.at<float>(0)=measurement.x;
                kf.statePost.at<float>(1)=measurement.y;
                kf.statePost.at<float>(2)=0;
                kf.statePost.at<float>(3)=0;
            }
            cv::circle(frame,measurement,radius,cv::Scalar(0,255,0),2);
            cv::circle(frame,predict,5,cv::Scalar(0,0,255),-1);
        }
        cv::imshow("predict",frame);
        if(cv::waitKey(30)==27)break;
        
    }

    return 0;
}