#include "paramLoader.h"

using namespace param;

paramHandler::paramHandler(const std::string settingPath)
{
    //-- 检查是否正确读取yaml文件
    cv::FileStorage fsSettings(settingPath.c_str(), cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
       std::cerr << "Failed to open settings file at: " << settingPath << std::endl;
       exit(-1);
    }
    std::cout << "\033[36m [paramHandler] \033[0m"
              << ": loading parameters from file "<< settingPath<<"..."<< std::endl;
    
    //-- 读取数据集类型与目录
    fsSettings["dataset"] >> dataset_type;
    fsSettings["pathDataset"] >> dataset_dir;


    //-- 读取相机参数
    fx = fsSettings["Camera.fx"];
    fy = fsSettings["Camera.fy"];
    cx = fsSettings["Camera.cx"];
    cy = fsSettings["Camera.cy"];

    //-- 构造畸变多项式
    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fsSettings["Camera.k1"];
    DistCoef.at<float>(1) = fsSettings["Camera.k2"];
    DistCoef.at<float>(2) = fsSettings["Camera.p1"];
    DistCoef.at<float>(3) = fsSettings["Camera.p2"];
    const float k3 = fsSettings["Camera.k3"];
    if(k3!=0)//-- 有些相机畸变系数中没有k3项
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }

    DistCoef.copyTo(mDistCoef);

    //-- 尺度因子要取倒数
    depth_scale = fsSettings["DepthMapFactor"];
    if(fabs(depth_scale)<1e-5)
        depth_scale=1;
    else
        depth_scale = 1.0f/depth_scale;
    
    //-- 打印参数用于检查
    std::cout<<"\033[36m [paramHandler] \033[0m"<<"- fx: " << fx << std::endl;
    std::cout<<"\033[36m [paramHandler] \033[0m"<<"- fy: " << fy << std::endl;
    std::cout<<"\033[36m [paramHandler] \033[0m"<<"- cx: " << cx << std::endl;
    std::cout<<"\033[36m [paramHandler] \033[0m"<<"- cy: " << cy << std::endl;
    std::cout<<"\033[36m [paramHandler] \033[0m"<<"- k1: " 
             << DistCoef.at<float>(0) << std::endl;
    std::cout<<"\033[36m [paramHandler] \033[0m"<<"- k2: " 
             << DistCoef.at<float>(1) << std::endl;
    std::cout<<"\033[36m [paramHandler] \033[0m"<<"- k3: " 
             << DistCoef.at<float>(4) << std::endl;
    std::cout<<"\033[36m [paramHandler] \033[0m"<<"- p1: " 
             << DistCoef.at<float>(2) << std::endl;
    std::cout<<"\033[36m [paramHandler] \033[0m"<<"- p2: "
             << DistCoef.at<float>(3) << std::endl;

    //-- 读取coarse部分的参数
    coarse.kf_rot_thres = fsSettings["coarse.kf_rot_thres"];
    coarse.kf_trans_thres = fsSettings["coarse.kf_trans_thres"];
    coarse.sample_bias = fsSettings["coarse.sample_bias"];
    coarse.maximum_point = fsSettings["coarse.maximum_point"];
    coarse.canny_high = fsSettings["coarse.cannyHigh"];
    coarse.canny_low = fsSettings["coarse.cannyLow"];

    std::cout<<"\033[36m [paramHandler] \033[0m"<<"\033[33m[coarse] \033[0m" 
             <<"- kf_rot_thres: " << coarse.kf_rot_thres<<std::endl;
    std::cout<<"\033[36m [paramHandler] \033[0m"<<"\033[33m[coarse] \033[0m" 
             <<"- kf_trans_thres: " << coarse.kf_trans_thres<<std::endl;
    std::cout<<"\033[36m [paramHandler] \033[0m"<<"\033[33m[coarse] \033[0m" 
             <<"- sample_bias: " << coarse.sample_bias<<std::endl;
    std::cout<<"\033[36m [paramHandler] \033[0m"<<"\033[33m[coarse] \033[0m" 
             <<"- maximum_point: " << coarse.maximum_point<<std::endl;
    std::cout<<"\033[36m [paramHandler] \033[0m"<<"\033[33m[coarse] \033[0m" 
             <<"- cannyHigh: " << coarse.canny_high<<std::endl;
    std::cout<<"\033[36m [paramHandler] \033[0m"<<"\033[33m[coarse] \033[0m" 
             <<"- cannyLow: " << coarse.canny_low<<std::endl;

    
    //-- 读取fine部分的参数
    fine.geo_photo_ratio = fsSettings["fine.geo_photo_ratio"];
    fine.kf_rot_thres = fsSettings["fine.kf_rot_thres"];
    fine.kf_trans_thres = fsSettings["fine.kf_trans_thres"];
    fine.sample_bias = fsSettings["fine.sample_bias"];
    fine.canny_high = fsSettings["fine.cannyHigh"];
    fine.canny_low = fsSettings["fine.cannyLow"];

    std::cout<<"\033[36m [paramHandler] \033[0m"<<"\033[32m[fine] \033[0m" 
             <<"- geo_photo_ratio: " << fine.geo_photo_ratio<<std::endl;
    std::cout<<"\033[36m [paramHandler] \033[0m"<<"\033[32m[fine] \033[0m" 
             <<"- kf_rot_thres: " << fine.kf_rot_thres<<std::endl;
    std::cout<<"\033[36m [paramHandler] \033[0m"<<"\033[32m[fine] \033[0m" 
             <<"- kf_trans_thres: " << fine.kf_trans_thres<<std::endl;
    std::cout<<"\033[36m [paramHandler] \033[0m"<<"\033[32m[fine] \033[0m" 
             <<"- sample_bias: " << fine.sample_bias<<std::endl;
    std::cout<<"\033[36m [paramHandler] \033[0m"<<"\033[32m[fine] \033[0m" 
             <<"- cannyHigh: " << fine.canny_high<<std::endl;
    std::cout<<"\033[36m [paramHandler] \033[0m"<<"\033[32m[fine] \033[0m" 
             <<"- cannyLow: " << fine.canny_low<<std::endl;

}