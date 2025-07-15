/**
 * @brief 几何配准计算残差与雅克比的类样例
 * @details 提供在3D空间中的点-点，点-线，以及点-面的几何配准的残差雅克比计算
 *          需要与先前完成的最小二乘类搭配使用
 * @date 2024-10-20
 * @author 猎魔人
 */
#ifndef GEOMETRYFACTOR_H
#define GEOMETRYFACTOR_H
#include <iostream>
#include <vector>
#include <ctime>
#include <algorithm>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <random>

//dependency for pcl
#include <pcl/console/parse.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>

#include <sophus/se3.hpp>

namespace eslam_core{

typedef Eigen::Matrix<float,  6, 6> Mat66f;
typedef Eigen::Matrix<double, 6, 6> Mat66d;
typedef Eigen::Matrix<float,  6, 1> Vec6f;
typedef Eigen::Matrix<double, 6, 1> Vec6d;

//将3维向量扩展成反对称矩阵
Eigen::Matrix3d hat(Eigen::Vector3d vec);
//罗德里格斯公式
Eigen::Matrix3d RodriguesFormular(Eigen::Vector3d r);

//构建SE(3)变换矩阵
Eigen::Matrix4d ConvertToSE3(Eigen::Matrix3d R, Eigen::Vector3d T);

//由六自由度向量（3平移3旋转）构造SE(3)变换矩阵
Eigen::Matrix4d ConvertToSE3(Eigen::Matrix<double , 6, 1> dx);

//-- 6dof的位姿向量做se3指数映射
Eigen::Matrix4d ExponentialSE3(Eigen::Matrix<double , 6, 1> dx);

//-- 将一个4x4的位姿矩阵转换为sophus::SE3d形式
Sophus::SE3d ConvertTrans2Sophus(Eigen::Matrix4d Pose);

//归一化线段的端点，将线段表示为 x = x0 + at, 取t=1时就能够有x0与x之间长度为1
std::pair<Eigen::Vector3d, Eigen::Vector3d> 
normalizeLinePoints(Eigen::Vector3d startPointRaw, Eigen::Vector3d endPointRaw);

//在2D空间（图像坐标系）归一化线段的端点，将线段表示为 x = x0 + at, 取t=1时就能够有x0与x之间长度为1
std::pair<Eigen::Vector2d, Eigen::Vector2d> 
normalizeLinePoints2D(Eigen::Vector2d startPointRaw, Eigen::Vector2d endPointRaw);

//-- 根据3D空间的点线残差计算雅克比矩阵，线由两个点描述
void calculateJacobiPointLine(Eigen::Vector3d startPoint, Eigen::Vector3d endPoint, Eigen::Vector3d queryPoint,
                              Eigen::Matrix3d R, Eigen::Vector3d t, 
                              Mat66d& H_out, Vec6d& g_out, double& residual);

//-- 根据3D-2D投影的点线残差计算雅克比矩阵，线由两个点描述
void calculateJacobiPointLine3D2D(Eigen::Vector2d startPoint, Eigen::Vector2d endPoint, Eigen::Vector3d queryPoint_3d,
                                Eigen::Matrix3d R, Eigen::Vector3d t, double fx, double fy, double cx, double cy,
                                Mat66d& H_out, Vec6d& g_out, double& residual);

void calculateJacobiPointLine3D2DVec(Eigen::Vector2d startPoint, Eigen::Vector2d endPoint, Eigen::Vector3d queryPoint_3d,
    Eigen::Matrix3d R, Eigen::Vector3d t, double fx, double fy, double cx, double cy,
    Mat66d& H_out, Vec6d& g_out, Eigen::Vector2d& residual);

}//-- namespace eslam_core

#endif