#ifndef EDGESELECTOR_H
#define EDGESELECTOR_H
#include <iostream>
#include <fstream>
#include <queue>
#include <stack>
#include <map>

#include <tbb/tbb.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include <chrono>

#include "edge.h"
#include "edgeCluster.h"

class edgeSelector{
public:
    //--原始数据
    cv::Mat mMatRGB;
    cv::Mat mMatGray;
    int mWidth;
    int mHeight;

    //-- 图像梯度的结果
    cv::Mat mMatGradMagnitude;
    cv::Mat mMatGradAngle;
    //-- 图像Canny算子计算得到的结果
    cv::Mat mMatCanny;

    //-- canny 算子的大小参数
    bool mbUseFixedThreshold; //-- true就使用固定阈值，false就使用适应阈值
    //-- 固定阈值时的固定参数
    int mpCanny_lower_bound;
    int mpCanny_higher_bound;

    //-- 属于同一个边缘的角度变化阈值（单位 degree）
    float mpAngle_bias;

    //-- 使用区域生长得到的边缘点列表
    std::vector<EdgeCluster> mvEdgeClusters;

    //-- 由零散点构造的有序边缘列表
    std::vector<Edge> mvEdges;

    /**
     * @brief [构造函数]固定Canny阈值来提取边缘
    */
    edgeSelector(float angle_bias, int canny_low, int canny_high)
    {
        //-- 启用 OpenCV 的并行优化
        cv::setUseOptimized(true);
        cv::setNumThreads(0);

        mpAngle_bias = angle_bias;
        mpCanny_higher_bound = canny_high;
        mpCanny_lower_bound = canny_low;
        mbUseFixedThreshold = true;
    }

    void processImage(const cv::Mat& image);

    cv::Mat visualizeEdges();

    cv::Mat visualizeEdgesOrganized();

private:

    float calcAngleBias(float angle_1, float angle_2);

    void regionGrowthClusteringOCanny(float angle_Thres);

    void preprocessCannyMat();

    void cvt2OrderedEdges();

    void cvt2OrderedEdgesParallel();
        
};

#endif