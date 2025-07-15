#include "geometryFactor.h"
// #define __DEBUG_STDOUT__
using namespace eslam_core;

//-- 将3维向量扩展成反对称矩阵
Eigen::Matrix3d eslam_core::hat(Eigen::Vector3d vec){
    Eigen::Matrix3d mat;
    mat<<        0, -vec.z(),  vec.y(),
           vec.z(),        0, -vec.x(),
          -vec.y(),  vec.x(),        0;
    return mat;
}

//-- 罗德里格斯公式
Eigen::Matrix3d eslam_core::RodriguesFormular(Eigen::Vector3d r)
{
    double theta = r.norm();
    r = r/theta;
    //-- 以下两个罗德里格斯公式等价
    // Eigen::Matrix3d R = cos(theta)*Eigen::Matrix3d::Identity() + 
    //                     (1-cos(theta))*r*r.transpose() + sin(theta)*hat(r);
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity() + (1-cos(theta)) * hat(r) * hat(r) + sin(theta)*hat(r);
    return R;
}

//-- 根据R,t构建SE(3)变换矩阵
Eigen::Matrix4d eslam_core::ConvertToSE3(Eigen::Matrix3d R, Eigen::Vector3d T)
{
    Eigen::Matrix4d Tr_44;
    Tr_44 << R(0,0), R(0,1), R(0,2), T(0),
             R(1,0), R(1,1), R(1,2), T(1),
             R(2,0), R(2,1), R(2,2), T(2),
             0,      0,      0,      1;
    return Tr_44;
}

//-- 由六自由度向量（3平移3旋转）构造SE(3)变换矩阵
Eigen::Matrix4d eslam_core::ExponentialSE3(Eigen::Matrix<double , 6, 1> dx){
    Eigen::Vector3d t_vec(dx(0), dx(1), dx(2));
    Eigen::Vector3d r_vec(dx(3), dx(4), dx(5));
    double theta = r_vec.norm();
    r_vec = r_vec/theta;
    //-- 罗德里格斯公式计算旋转
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity() + (1-cos(theta)) * hat(r_vec) * hat(r_vec) + sin(theta)*hat(r_vec);
    //-- se3同样需要做指数映射的平移
    Eigen::Matrix3d J = sin(theta)/theta * Eigen::Matrix3d::Identity() + 
                        (1 - sin(theta)/theta)*r_vec*r_vec.transpose() + 
                        (1-cos(theta))/theta*hat(r_vec);
    Eigen::Matrix4d T = Eigen::MatrixXd::Identity(4,4);
    T.block(0,3,3,1) = J * t_vec;
    T.block(0,0,3,3) = R;
    return T;
}

Sophus::SE3d eslam_core::ConvertTrans2Sophus(Eigen::Matrix4d Pose)
{
    Eigen::Matrix3d R = Pose.block(0,0,3,3);
    Eigen::Vector3d t = Pose.block(0,3,3,1);
    Eigen::Quaterniond q(R);
    q.normalize();
    Sophus::SE3d pose(q,t);
    return pose;
}

Eigen::Matrix4d eslam_core::ConvertToSE3(Eigen::Matrix<double , 6, 1> dx){
    Eigen::Vector3d translation(dx(0), dx(1), dx(2));
    Eigen::Vector3d rotationVec(dx(3), dx(4), dx(5));
    Eigen::Matrix3d R = RodriguesFormular(rotationVec);
    Eigen::Matrix4d T = ConvertToSE3(R, translation);
    return T;
}

std::pair<Eigen::Vector3d, Eigen::Vector3d> 
eslam_core::normalizeLinePoints(Eigen::Vector3d startPointRaw, Eigen::Vector3d endPointRaw)
{
    Eigen::Vector3d dirRaw = endPointRaw - startPointRaw;
    Eigen::Vector3d dirNorm = dirRaw/dirRaw.norm();
    Eigen::Vector3d startPoint = startPointRaw;
    Eigen::Vector3d endPoint = startPoint + dirNorm;
    std::pair<Eigen::Vector3d, Eigen::Vector3d> linePair;
    linePair.first = startPoint; linePair.second = endPoint;
    return linePair;
}

std::pair<Eigen::Vector2d, Eigen::Vector2d> 
eslam_core::normalizeLinePoints2D(Eigen::Vector2d startPointRaw, Eigen::Vector2d endPointRaw)
{
    Eigen::Vector2d dirRaw = endPointRaw - startPointRaw;
    Eigen::Vector2d dirNorm = dirRaw/dirRaw.norm();
    Eigen::Vector2d startPoint = startPointRaw;
    Eigen::Vector2d endPoint = startPoint + dirNorm;
    std::pair<Eigen::Vector2d, Eigen::Vector2d> linePair;
    linePair.first = startPoint; linePair.second = endPoint;
    return linePair;
}

void eslam_core::calculateJacobiPointLine(Eigen::Vector3d startPoint, Eigen::Vector3d endPoint, Eigen::Vector3d queryPoint,
                                          Eigen::Matrix3d R, Eigen::Vector3d t, 
                                          Mat66d& H_out, Vec6d& g_out, double& residual)
{
    //-- normalize border points.
    std::pair<Eigen::Vector3d, Eigen::Vector3d> normPoints = normalizeLinePoints(startPoint,endPoint);
    
    //-- defination of registrated line.
    Eigen::Vector3d a = normPoints.first;
    Eigen::Vector3d b = normPoints.second;
    Eigen::Vector3d l_ba = a-b;

    Mat66d H = Eigen::Matrix<double , 6, 6>::Zero();
    Vec6d  g = Eigen::Matrix<double , 6, 1>::Zero();

    Eigen::Vector3d transed_point = R * queryPoint + t;
    
    //-- 3D位姿变换产生的3D位姿残差的雅克比矩阵为3x6
    Eigen::Matrix<double , 3, 6> J = Eigen::Matrix<double , 3, 6>::Zero();
    
    //-- 图示计算残差的三角形，残差向量为l_dc，c为位姿变换后的query point
    //    *a----------*d---*b
    //      ---       |   /
    //         ---    |  /
    //            --- | /
    //                *c
    //-- dr/dt = d(l_dc)/dt = d(l_bc)/dt - d(l_bd)/dt
    //--  l_bd = l_ba * (l_ba^T . l_bc), l_bc = pc - pb = R*q + t - p_b
    //-- dr/dt = I - l_ba * l_ba^T
    J.block(0,0,3,3) = Eigen::Matrix3d::Identity() - l_ba * l_ba.transpose();
    
    //-- 对旋转的雅克比矩阵, 形式上与平移相同，只是多一个Rq+t对旋转的求导
    //--      dr/dR = d(l_dc)/dR = d(l_bc)/dR - d(l_bd)/dR
    //-- d(l_bc)/dR = -Rq^
    //-- d(l_bd)/dR = d[l_ba * (l_ba^T . l_bc)]/dR = -l_ba * l_ba^T * Rq^
    J.block(0,3,3,3) = -hat(transed_point) + l_ba * l_ba.transpose() * hat(transed_point);
    
    //-- 计算残差 r = l_bc - l_bd
    Eigen::Vector3d l_bc = transed_point - b;
    Eigen::Vector3d l_bd = l_ba * (l_ba.transpose() * l_bc);
    Eigen::Vector3d r = l_bc - l_bd;

    //-- update derivative matrix
    residual = r.squaredNorm();
    g = -J.transpose() * r; g_out = g;
    H = J.transpose() * J;  H_out = H;
}

void eslam_core::calculateJacobiPointLine3D2D(Eigen::Vector2d startPoint, Eigen::Vector2d endPoint, Eigen::Vector3d queryPoint_3d,
                                            Eigen::Matrix3d R, Eigen::Vector3d t, double fx, double fy, double cx, double cy,
                                            Mat66d& H_out, Vec6d& g_out, double& residual)
{
    //-- normalize border points.
    std::pair<Eigen::Vector2d, Eigen::Vector2d> normPoints = normalizeLinePoints2D(startPoint,endPoint);
    
    //-- defination of registrated line.
    Eigen::Vector2d a = normPoints.first;
    Eigen::Vector2d b = normPoints.second;
    Eigen::Vector2d l_ba = a-b;

    Mat66d H = Eigen::Matrix<double , 6, 6>::Zero();
    Vec6d  g = Eigen::Matrix<double , 6, 1>::Zero();

    //-- 计算3D空间的位姿变换
    Eigen::Vector3d transed_point = R * queryPoint_3d + t;
    //-- 计算重投影坐标
    double inv_z = 1.0 / transed_point[2];
    double inv_z2 = inv_z * inv_z;
    Eigen::Vector2d proj(fx * transed_point[0] / transed_point[2] + cx, 
                         fy * transed_point[1] / transed_point[2] + cy);

    
    //-- 3D位姿变换产生的3D-2D投影的位姿残差的雅克比矩阵为2x6
    Eigen::Matrix<double , 2, 6> J_orig = Eigen::Matrix<double , 2, 6>::Zero();
    Eigen::Matrix<double , 2, 6> J = Eigen::Matrix<double , 2, 6>::Zero();
    //-- 一般重投影误差的2x6残差矩阵, fx/z如果是负的说明这是 关联像素-重投影像素，正的则是 重投影像素-关联像素
    J_orig << -fx * inv_z,
        0,
        fx * transed_point[0] * inv_z2,
        fx * transed_point[0] * transed_point[1] * inv_z2,
        -fx - fx * transed_point[0] * transed_point[0] * inv_z2,
        fx * transed_point[1] * inv_z,
        0,
        -fy * inv_z,
        fy * transed_point[1] * inv_z,
        fy + fy * transed_point[1] * transed_point[1] * inv_z2,
        -fy * transed_point[0] * transed_point[1] * inv_z2,
        -fy * transed_point[0] * inv_z;
    //-- 由于根据下面的计算是l_dc是残差故而J_orig要反一下
    J_orig = -J_orig;
    Eigen::Matrix<double , 2, 3> J_orig_t = J_orig.block(0,0,2,3);
    Eigen::Matrix<double , 2, 3> J_orig_R = J_orig.block(0,3,2,3);
    
    //-- 图示计算残差的三角形，残差向量为l_dc，c为位姿变换后的query point
    //    *a----------*d---*b
    //      ---       |   /
    //         ---    |  /
    //            --- | /
    //                *c
    //-- dr/dt = d(l_dc)/dt = d(l_bc)/dt - d(l_bd)/dt
    //--  l_bd = l_ba * (l_ba^T . l_bc), l_bc = pc - pb = warp(q) - p_b
    //-- dr/dt = (I - l_ba * l_ba^T) J_3d2d

    //-- 点线残差构造时的此项都是固定的一个2x2的倍数矩阵：(I - l_ba * l_ba^T)
    Eigen::Matrix<double, 2, 2> k_line = Eigen::Matrix<double , 2, 2>::Zero();
    k_line = Eigen::Matrix2d::Identity() - l_ba * l_ba.transpose();

    //-- 平移部分的量
    J.block(0,0,2,3) = k_line * J_orig_t;
    
    //-- 对旋转的雅克比矩阵, 形式上与平移相同，只是多一个Rq+t对旋转的求导
    //--      dr/dR = d(l_dc)/dR = d(l_bc)/dR - d(l_bd)/dR
    //-- d(l_bc)/dR = d warp(q)/dR
    //-- d(l_bd)/dR = d[l_ba * (l_ba^T . l_bc)]/dR = -l_ba * l_ba^T * Rq^
    J.block(0,3,2,3) = k_line * J_orig_R;
    
    //-- 计算残差 r = l_bc - l_bd
    Eigen::Vector2d l_bc = proj - b;
    Eigen::Vector2d l_bd = l_ba * (l_ba.transpose() * l_bc);
    Eigen::Vector2d r = l_bc - l_bd;

    //-- update derivative matrix
    residual = r.squaredNorm();
    g = -J.transpose() * r; g_out = g;
    H = J.transpose() * J;  H_out = H;
}

void eslam_core::calculateJacobiPointLine3D2DVec(Eigen::Vector2d startPoint, Eigen::Vector2d endPoint, Eigen::Vector3d queryPoint_3d,
                                            Eigen::Matrix3d R, Eigen::Vector3d t, double fx, double fy, double cx, double cy,
                                            Mat66d& H_out, Vec6d& g_out, Eigen::Vector2d& residual)
{
    //-- normalize border points.
    std::pair<Eigen::Vector2d, Eigen::Vector2d> normPoints = normalizeLinePoints2D(startPoint,endPoint);
    
    //-- defination of registrated line.
    Eigen::Vector2d a = normPoints.first;
    Eigen::Vector2d b = normPoints.second;
    Eigen::Vector2d l_ba = a-b;

    Mat66d H = Eigen::Matrix<double , 6, 6>::Zero();
    Vec6d  g = Eigen::Matrix<double , 6, 1>::Zero();

    //-- 计算3D空间的位姿变换
    Eigen::Vector3d transed_point = R * queryPoint_3d + t;
    //-- 计算重投影坐标
    double inv_z = 1.0 / transed_point[2];
    double inv_z2 = inv_z * inv_z;
    Eigen::Vector2d proj(fx * transed_point[0] / transed_point[2] + cx, 
                         fy * transed_point[1] / transed_point[2] + cy);

    
    //-- 3D位姿变换产生的3D-2D投影的位姿残差的雅克比矩阵为2x6
    Eigen::Matrix<double , 2, 6> J_orig = Eigen::Matrix<double , 2, 6>::Zero();
    Eigen::Matrix<double , 2, 6> J = Eigen::Matrix<double , 2, 6>::Zero();
    //-- 一般重投影误差的2x6残差矩阵, fx/z如果是负的说明这是 关联像素-重投影像素，正的则是 重投影像素-关联像素
    J_orig << -fx * inv_z,
        0,
        fx * transed_point[0] * inv_z2,
        fx * transed_point[0] * transed_point[1] * inv_z2,
        -fx - fx * transed_point[0] * transed_point[0] * inv_z2,
        fx * transed_point[1] * inv_z,
        0,
        -fy * inv_z,
        fy * transed_point[1] * inv_z,
        fy + fy * transed_point[1] * transed_point[1] * inv_z2,
        -fy * transed_point[0] * transed_point[1] * inv_z2,
        -fy * transed_point[0] * inv_z;
    //-- 由于根据下面的计算是l_dc是残差故而J_orig要反一下
    J_orig = -J_orig;
    Eigen::Matrix<double , 2, 3> J_orig_t = J_orig.block(0,0,2,3);
    Eigen::Matrix<double , 2, 3> J_orig_R = J_orig.block(0,3,2,3);
    
    //-- 图示计算残差的三角形，残差向量为l_dc，c为位姿变换后的query point
    //    *a----------*d---*b
    //      ---       |   /
    //         ---    |  /
    //            --- | /
    //                *c
    //-- dr/dt = d(l_dc)/dt = d(l_bc)/dt - d(l_bd)/dt
    //--  l_bd = l_ba * (l_ba^T . l_bc), l_bc = pc - pb = warp(q) - p_b
    //-- dr/dt = (I - l_ba * l_ba^T) J_3d2d

    //-- 点线残差构造时的此项都是固定的一个2x2的倍数矩阵：(I - l_ba * l_ba^T)
    Eigen::Matrix<double, 2, 2> k_line = Eigen::Matrix<double , 2, 2>::Zero();
    k_line = Eigen::Matrix2d::Identity() - l_ba * l_ba.transpose();

    //-- 平移部分的量
    J.block(0,0,2,3) = k_line * J_orig_t;
    
    //-- 对旋转的雅克比矩阵, 形式上与平移相同，只是多一个Rq+t对旋转的求导
    //--      dr/dR = d(l_dc)/dR = d(l_bc)/dR - d(l_bd)/dR
    //-- d(l_bc)/dR = d warp(q)/dR
    //-- d(l_bd)/dR = d[l_ba * (l_ba^T . l_bc)]/dR = -l_ba * l_ba^T * Rq^
    J.block(0,3,2,3) = k_line * J_orig_R;
    
    //-- 计算残差 r = l_bc - l_bd
    Eigen::Vector2d l_bc = proj - b;
    Eigen::Vector2d l_bd = l_ba * (l_ba.transpose() * l_bc);
    Eigen::Vector2d r = l_bc - l_bd;

    //-- update derivative matrix
    residual = r;
    g = -J.transpose() * r; g_out = g;
    H = J.transpose() * J;  H_out = H;
}