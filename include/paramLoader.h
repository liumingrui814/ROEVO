#ifndef PARAM_LOADER_H
#define PARAM_LOADER_H

#include <iostream>
#include <fstream>
#include <queue>
#include <sys/types.h>
#include <dirent.h>
#include <map>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

namespace param{

struct coarseParam{
    //-- 选取关键帧的平移与旋转阈值
    double kf_trans_thres;
    double kf_rot_thres;

    //-- 采样时的间隔距离
    int sample_bias;
    
    //-- 采样时允许的最大点的间隔
    int maximum_point;

    int canny_high;
    int canny_low;
};

struct fineParam{
    //-- 选取关键帧的平移与旋转阈值
    double kf_trans_thres;
    double kf_rot_thres;

    //-- 几何残差与光度残差的量纲比例
    double geo_photo_ratio;

    //-- 均匀采样的间隔点数
    int sample_bias;

    int canny_high;
    int canny_low;
};

class paramHandler{
public:

    std::string dataset_type;
    std::string dataset_dir;

    double fx;
    double fy;
    double cx;
    double cy;
    double depth_scale;

    coarseParam coarse;
    fineParam fine;

    cv::Mat mDistCoef;

    paramHandler(const std::string settingPath);

};

}
#endif