#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "KeyFrame.h"
#include "disjointSet.h"
#include "elementEdge.h"
#include "localMap.h"
#include "fineTracker.h"

namespace edge_map{

class Optimizer{
public:

    // 禁止实例化
    Optimizer() = delete;

    //-- 优化一帧关键帧与局部地图的相对位姿（局部地图3D--关键帧2D）
    static void optimizeSingleKFRef2Cur(const KeyFramePtr& pKF, std::vector<match3d_2d> matches, 
                                        std::vector<double> weights, Sophus::SE3d& adjust_pose);

    //-- 优化一帧关键帧与局部地图的相对位姿（关键帧3D--局部地图2D）
    static void optimizeSingleKFCur2Ref(const KeyFramePtr& pKF, std::vector<match3d_2d> matches, 
                                        const Sophus::SE3d& pose_ref, Sophus::SE3d& adjust_pose);

    //-- 优化局部地图中所有关键帧相对于局部地图的相对位姿
    static void optimizeAllInvolvedKFs(const localMapPtr pLocalMap);

    //-- Bundle Adjustment

    //-- 调整整个窗口内关键帧的全局位姿

private:

    static void RegistrationGeometricParallel(std::vector<Eigen::Vector3d> vGeometricPoints,
        std::vector<float> vScoreDepths,
        std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> vAssociatedLines,
        Sophus::SE3d& poseKF, float fx, float fy, float cx, float cy);

    static void calculateJacobiPointLine3D2D(Eigen::Vector2d startPoint, Eigen::Vector2d endPoint, Eigen::Vector3d queryPoint_3d,
        fine::Mat66d& H_out, fine::Vec6d& g_out, Eigen::Vector2d& residual, Sophus::SE3d& pose, float fx, float fy, float cx, float cy);

    static std::pair<Eigen::Vector2d, Eigen::Vector2d> 
        normalizeLinePoints2D(Eigen::Vector2d startPointRaw, Eigen::Vector2d endPointRaw);

};

}

#endif