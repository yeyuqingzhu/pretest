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
    cv::GaussianBlur(Input,Blur,cv::Size(5,5),0);
    cv::Moments m=cv::moments(Blur,true);
    if(m.m00<100)return false;
    point.x=m.m10/m.m00;
    point.y=m.m01/m.m00;
    radius=std::sqrt(m.m00/CV_PI);
    return true;
}

//卡尔曼滤波器
cv::Point2f KalmanTrack(cv::KalmanFilter& kf,const cv::Point2f& measurement,bool found,bool& initialized)
{
    static cv::Point2f last_measurement;

    // 1. 预测
    cv::Mat prediction=kf.predict();

    // 2. 如果检测到目标
    if(found)
    {
        if(!initialized)
        {
            // 初始化位置
            kf.statePost.at<float>(0)=measurement.x;
            kf.statePost.at<float>(1)=measurement.y;

            // 初始化速度（用上一帧估计）
            kf.statePost.at<float>(2)=measurement.x - last_measurement.x;
            kf.statePost.at<float>(3)=measurement.y - last_measurement.y;

            // 初始化加速度
            kf.statePost.at<float>(4)=0;
            kf.statePost.at<float>(5)=0;

            initialized = true;
        }
        else
        {
            // 构造测量向量
            cv::Mat meas(2,1,CV_32F);
            meas.at<float>(0)=measurement.x;
            meas.at<float>(1)=measurement.y;

            // 卡尔曼校正
            kf.correct(meas);
        }

        last_measurement=measurement;
    }

    // 3. 返回当前状态（校正后的）
    cv::Point2f state;
    state.x=kf.statePost.at<float>(0);
    state.y=kf.statePost.at<float>(1);

    float vx=kf.statePost.at<float>(2);
    float vy=kf.statePost.at<float>(3);

    //预测强度
    float PredictGain=0.5;
    
    //前向预测
    state.x+=vx*PredictGain;
    state.y+=vy*PredictGain;

    return state;
}
//---------------------------main-------------------------------------------
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
    int num=0,LostFrames=0;
    const int MAX_LOST=10;
    double radius;

    //设置视频参数
    int fps=cap.get(cv::CAP_PROP_FPS);
    int width=cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height=cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    std::string filename="output.mp4";

    //定义编码格式 (FourCC)
    int fourcc=cv::VideoWriter::fourcc('m','p','4','v'); // 对应 mp4v

    //创建 VideoWriter 对象
    cv::VideoWriter writer;
    writer.open(filename,fourcc,fps,cv::Size(width,height));

    //检查是否成功打开
    if (!writer.isOpened()){
        std::cerr<<"错误：无法创建视频文件！"<<std::endl;
        return -1;
    }

    //-----------------卡尔曼滤波---------------------
        cv::KalmanFilter kf(4,2,0);

        double dt=1.0f/fps;
        // 状态转移矩阵
        kf.transitionMatrix=(cv::Mat_<float>(4,4)<<
        1,0,dt,0, 
        0,1,0 ,dt,
        0,0,1 ,0 ,
        0,0,0 ,1 );

        // 测量矩阵
        kf.measurementMatrix=cv::Mat::zeros(2,4,CV_32F);
        kf.measurementMatrix.at<float>(0,0)=1;
        kf.measurementMatrix.at<float>(1,1)=1;

        // 噪声矩阵
        setIdentity(kf.processNoiseCov,cv::Scalar::all(5));
        setIdentity(kf.measurementNoiseCov,cv::Scalar::all(0.5));
        setIdentity(kf.errorCovPost,cv::Scalar::all(1));

        kf.statePost=cv::Mat::zeros(4,1,CV_32F);

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

        std::cout<<found;

        predict=KalmanTrack(kf,measurement,found,firstfound);

        //绘制并显示结果
        if(found){
            LostFrames=0;

            cv::circle(frame,measurement,radius,cv::Scalar(0,255,0),2);
            cv::circle(frame,predict,radius,cv::Scalar(0,0,255),2);
        }else{
            LostFrames++;
        }
        if(LostFrames>=MAX_LOST)firstfound=0;
        writer.write(frame);
        cv::imshow("predict",frame);
        if(cv::waitKey(30)==27)break;

    }

    //释放资源
    writer.release();

    std::cout<<"视频已保存为"<<filename<<std::endl;

    return 0;
}