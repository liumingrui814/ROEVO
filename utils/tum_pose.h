#ifndef TUM_POSE_H
#define TUM_POSE_H
#include <iomanip> // 注意包含这个头文件, 有效数字需要
#include <iostream>
#include <fstream>
#include <vector>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
namespace tum_pose{

//-- 位姿插值，返回TransFront与TransLast之间的差值位姿的proportion比例的位姿
Eigen::Matrix4d PoseInterpolation(Eigen::Matrix4d TransFront, Eigen::Matrix4d TransLast, double proportion)
{
    Eigen::Matrix4d TransBias = TransFront.inverse() * TransLast;
    Eigen::Vector3d translation;
    translation(0)=TransBias(0,3)*proportion;
    translation(1)=TransBias(1,3)*proportion;
    translation(2)=TransBias(2,3)*proportion;
    Eigen::Matrix3d R = TransBias.block(0,0,3,3);
    Eigen::Quaterniond q(R);
    q = Eigen::Quaterniond::Identity().slerp(proportion, q);
    R = q.toRotationMatrix();
    TransBias<<R(0,0), R(0,1), R(0,2), translation.x(),
               R(1,0), R(1,1), R(1,2), translation.y(),
               R(2,0), R(2,1), R(2,2), translation.z(),
                    0,      0,      0,               1;
    return TransBias;
}


Eigen::Matrix4d getStaticGTPose(double obsrvtime, 
                                std::vector<double> vector_stamp,
                                std::vector<Eigen::Matrix4d> vector_trans)
{
    //-- NaN值相当于一个无效矩阵
    Eigen::Matrix4d NaN = Eigen::MatrixXd::Zero(4,4);
    int cnt = 0;
    while(vector_stamp[cnt] < obsrvtime && cnt < vector_stamp.size()) cnt++;
    //-- 现在的cnt是第一个stamp大于等于sensorTime的index
    int PointerLast = cnt;     //-- 第一个比sensorTime大的时间戳的index
    int PointerFront = cnt-1;  //-- 第一个比sensorTime小的时间戳的index

    //-- 如果时间戳完全相等，则返回对应位姿
    if(fabs(obsrvtime - vector_stamp[PointerLast])<0.0001){
        return vector_trans[PointerLast];
    }

    if(PointerFront < 0 || PointerLast >= vector_stamp.size()){
        //-- gt列表中已经无法通过插值获取当前帧的位姿
        return NaN;
    }
    //-- 剩下的部分可以插值
    double lasttime  = vector_stamp[PointerLast];
    double fronttime = vector_stamp[PointerFront];
    //-- |<---- | ----------> |
    //-- front  curr          last
    //-- |<---->| obsrv_time - front_time
    //-- |<------------------>| last_time - front_time
    double proportion = (obsrvtime-fronttime)/(lasttime-fronttime);
    Eigen::Matrix4d TransLast  = vector_trans[PointerLast];
    Eigen::Matrix4d TransFront = vector_trans[PointerFront];
    Eigen::Matrix4d TransBias1 = PoseInterpolation(TransFront, TransLast, proportion);
    return TransFront * TransBias1;
}

bool getStaticGTPoseValid(double obsrvtime, 
                        std::vector<double> vector_stamp,
                        std::vector<Eigen::Matrix4d> vector_trans,
                        Eigen::Matrix4d& pose_gt)
{
    //-- NaN值相当于一个无效矩阵
    Eigen::Matrix4d NaN = Eigen::MatrixXd::Zero(4,4);
    int cnt = 0;
    while(vector_stamp[cnt] < obsrvtime && cnt < vector_stamp.size()) cnt++;
    //-- 现在的cnt是第一个stamp大于等于sensorTime的index
    int PointerLast = cnt;     //-- 第一个比sensorTime大的时间戳的index
    int PointerFront = cnt-1;  //-- 第一个比sensorTime小的时间戳的index

    //-- 如果时间戳完全相等，则返回对应位姿
    if(fabs(obsrvtime - vector_stamp[PointerLast])<0.0001){
        pose_gt = vector_trans[PointerLast];
        return true;
    }

    if(PointerFront < 0 || PointerLast >= vector_stamp.size()){
        //-- gt列表中已经无法通过插值获取当前帧的位姿
        pose_gt = NaN;
        return false;
    }

    //-- 判断前后两帧真值帧的时间差会不会太大太大（30ms以上就算很大）
    if(std::fabs(vector_stamp[PointerLast] - vector_stamp[PointerFront]) > 0.03){
        pose_gt = NaN;
        return false;
    }
    //-- 剩下的部分可以插值
    double lasttime  = vector_stamp[PointerLast];
    double fronttime = vector_stamp[PointerFront];
    //-- |<---- | ----------> |
    //-- front  curr          last
    //-- |<---->| obsrv_time - front_time
    //-- |<------------------>| last_time - front_time
    double proportion = (obsrvtime-fronttime)/(lasttime-fronttime);
    Eigen::Matrix4d TransLast  = vector_trans[PointerLast];
    Eigen::Matrix4d TransFront = vector_trans[PointerFront];
    Eigen::Matrix4d TransBias1 = PoseInterpolation(TransFront, TransLast, proportion);
    pose_gt = TransFront * TransBias1;
    return true;
}

bool getPoseBiasGT(double obsrvtime_1, double obsrvtime_2, 
                std::vector<double> vector_stamp,
                std::vector<Eigen::Matrix4d> vector_trans,
                Eigen::Matrix4d& pose_gt)
{
    Eigen::Matrix4d pose_from, pose_to;
    bool valid_1 = getStaticGTPoseValid(obsrvtime_1, vector_stamp, vector_trans, pose_from);
    bool valid_2 = getStaticGTPoseValid(obsrvtime_2, vector_stamp, vector_trans, pose_to);
    if(valid_1 && valid_2)
    {
        pose_gt = pose_from.inverse() * pose_to;
        return true;
    }else{
        return false;
    }
}


}

#endif