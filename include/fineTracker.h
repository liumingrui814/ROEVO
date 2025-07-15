#ifndef FINETRACKER_H
#define FINETRACKER_H

#include <iostream>
#include <fstream>
//dependency for opencv
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
//dependency for eigen
#include <Eigen/Core>
#include <Eigen/Dense>
//dependency for sophus
#include <sophus/se3.hpp>
//dependency for tbb multi-thread
#include <tbb/tbb.h>
#include <tbb/parallel_for.h>
#include <tbb/concurrent_vector.h>

#include <mutex>

#include "edgeSelector.h"
#include "Frame.h"
#include "robustWeight.h"
#include "disjointSet.h"

#include <chrono> //-- 计时函数

namespace fine{

typedef Eigen::Matrix<float,  6, 6> Mat66f;
typedef Eigen::Matrix<double, 6, 6> Mat66d;
typedef Eigen::Matrix<float,  6, 1> Vec6f;
typedef Eigen::Matrix<double, 6, 1> Vec6d;

class FineTracker {
    public:
        //-- 当类中包含固定大小的 Eigen 对象（如 Eigen::Vector2d、Eigen::Matrix4f 等）
        //-- 作为成员变量，并且类会通过 new 动态分配时，必须使用此宏
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    
        FineTracker(double fx, double fy, double cx, double cy, float ratio);
    
        //-- 设置参考帧，参考帧就是关键帧指针
        void setReference(const FramePtr kf_ref);
        //-- 设置当前帧，当前帧就是普通帧指针
        void setCurrent(const FramePtr f_curr);
        //-- 设置参考帧到当前帧的位姿先验
        void setPosePriorRef2Cur(const Sophus::SE3d& T);
        //-- 设置当前帧到参考帧的位姿先验
        void setPosePriorCur2Ref(const Sophus::SE3d& T);

        //-- 在给定参考帧与当前帧的情况下估计 ref->curr的位姿变换
        void estimate(Sophus::SE3d &T21, bool use_parallel = true);

        std::vector<orderedEdgePoint> getGeoPoints()
        {
            return mvGeometryPoints;
        }
    
    private:
        
        FramePtr mpF_ref;
        FramePtr mpF_cur;

        //-- 当前帧到参考帧的位姿变换，用 coarse tracking 的结果初始化，持续更新
        //-- 因为估计的是将参考帧的3D特征投影到当前帧，所以用 T_cur_ref, 最终要变成 T_ref_cur
        Sophus::SE3d T_cur_ref;

        float mFx, mFy, mCx, mCy;

        float geo_photo_ratio;

        //-- 存在 edge-wise correspondence 并且存在关联的点集
        std::vector<orderedEdgePoint> mvGeometryPoints;

        //-- mvGeometryPoints 关联得到的当前帧的 2D 线段
        std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> mvAssociatedLines;

        void associationRef2Cur();
        void associationRef2CurParallel();

        void getAssociationLines2D();

        void RegistrationCombined();
        void RegistrationCombinedParallel();

        void calculateJacobiPointLine3D2D(Eigen::Vector2d startPoint, Eigen::Vector2d endPoint, Eigen::Vector3d queryPoint_3d,
                                          Mat66d& H_out, Vec6d& g_out, Eigen::Vector2d& residual);

        float GetPixelValue(const cv::Mat &img, float x, float y);

        void calculateJacobiPhotometric(const cv::Mat& image_ref, const cv::Mat& image_cur,
                                        cv::Point queryPixel, float depth,
                                        Mat66d& H_out, Vec6d& g_out, double& cost_out);
    
        
        
    };


}//namespace fine


#endif