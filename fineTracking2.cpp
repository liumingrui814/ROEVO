/**
 * @brief 精匹配文件，根据粗匹配的结果引入数据关联，在数据关联的基础上进行point-tangent optimization以获得更好的结果
 * @details 粗匹配得到位姿结果后，精匹配的部分直接读取位姿并作为优化的初值
 * @author 猎魔人
 * @date 2024-10-23 添加基于点线（3D-2D）残差点云配准的优化
 * @details 参考论文/repository: 
*/
#include <iostream>
#include <fstream>
#include <sophus/se3.hpp>

#include <pcl/console/parse.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transforms.h>

#include "edgeSelector.h"
#include "Frame.h"
#include "utils/tum_pose.h"
#include "utils/tum_file.h"
#include "disjointSet.h"

#include "fineTracker.h"

#include <chrono> //-- 计时函数

#include "paramLoader.h"
// #define __DEBUG_OPTIMIZATION__

// typedef std::pair<std::vector<orderedEdgePoint>, std::vector<orderedEdgePoint>> assoResult;

//-- 本文件涉及的全局变量
float fx; 
float fy; 
float cx; 
float cy;
float kf_trans_thres;
float kf_rot_thres;
float geo_photo_ratio;
int sample_bias;

Sophus::SE3d convertTrans2Sophus(const Eigen::Matrix4d& T_ref_cur_coarse) {
    // 提取旋转部分 (3x3 左上角矩阵)
    Eigen::Matrix3d R = T_ref_cur_coarse.block<3, 3>(0, 0);
    
    // 提取平移部分 (3x1 右上角向量)
    Eigen::Vector3d t = T_ref_cur_coarse.block<3, 1>(0, 3);
    
    // 创建 Sophus SE3 对象
    // 注意: Sophus::SE3d 的构造函数需要先旋转后平移
    Sophus::SE3d SE3(R, t);
    
    return SE3;
}

cv::Mat visualizeReprojection(const std::vector<orderedEdgePoint> &pts_geo, 
                              const std::vector<orderedEdgePoint> &pts_pho, 
                              Eigen::Matrix4d pose_ref_cur, Frame frame_ref, cv::Mat img_backGround)
{
    Eigen::Matrix3d R = pose_ref_cur.block(0,0,3,3);
    Eigen::Vector3d t = pose_ref_cur.block(0,3,3,1);
    cv::Mat img_viz = img_backGround.clone();
    //-- 对优化的当前帧点云进行重投影，重投影回参考帧
    if(img_viz.channels()==1){
        cv::cvtColor(img_viz, img_viz, cv::COLOR_GRAY2BGR);
    }

    //-- 绘制几何残差的前后投影像素
    for(int i = 0; i < pts_geo.size(); ++i){
        cv::Point geo_pixel = cv::Point(pts_geo[i].x, pts_geo[i].y);
        //-- 几何的点才有可能存在像素点关联，因而这里可视化这种像素点关联
        if(pts_geo[i].mbAssociated == true){
            int asso_edge_ID = pts_geo[i].asso_edge_ID;
            int asso_point_index = pts_geo[i].asso_point_index;
            int ref_idx = frame_ref.mmIndexMap[asso_edge_ID];
            orderedEdgePoint pt_asso = frame_ref.mvEdges[ref_idx].mvPoints[asso_point_index];
            cv::Point asso_pixel(pt_asso.x, pt_asso.y);
            cv::line(img_viz, geo_pixel, asso_pixel, cv::Scalar(160,255,255), 1, cv::LINE_AA);
            cv::circle(img_viz, asso_pixel, 1, cv::Scalar(0,255,0), -1, cv::LINE_AA);
        }
        cv::circle(img_viz, geo_pixel, 1, cv::Scalar(0,0,255), -1, cv::LINE_AA);
    }

    //-- 绘制光度误差的前后重投影像素
    for(int i = 0; i < pts_pho.size(); ++i){
        cv::Point geo_pixel = cv::Point(pts_pho[i].x, pts_pho[i].y);
        //-- 绘制原本的当前帧像素
        cv::circle(img_viz, geo_pixel, 1, cv::Scalar(0,100,255), -1, cv::LINE_AA);
        //-- 计算重投影的像素并重新绘制重投影像素
        Eigen::Vector3d pt_3d(pts_pho[i].x_3d, pts_pho[i].y_3d, pts_pho[i].z_3d);
        Eigen::Vector3d transed_point = R * pt_3d + t;
        //-- 计算重投影坐标
        double inv_z = 1.0 / transed_point[2];
        double inv_z2 = inv_z * inv_z;
        Eigen::Vector2d proj(fx * transed_point[0] / transed_point[2] + cx, 
                            fy * transed_point[1] / transed_point[2] + cy);
        cv::circle(img_viz, cv::Point(proj(0), proj(1)), 1, cv::Scalar(0,255,255), -1, cv::LINE_AA);
    }

    return img_viz;
}

cv::Mat visualizeReprojectionBias(const std::vector<orderedEdgePoint> &pts_geo, 
                                  FramePtr frame_cur, cv::Mat img_backGround)
{
    cv::Mat img_viz = img_backGround.clone();
    //-- 对优化的当前帧点云进行重投影，重投影回参考帧
    if(img_viz.channels()==1){
        cv::cvtColor(img_viz, img_viz, cv::COLOR_GRAY2BGR);
    }

    //-- 绘制几何残差的前后投影像素
    for(int i = 0; i < pts_geo.size(); ++i){
        cv::Point geo_pixel = cv::Point(pts_geo[i].x, pts_geo[i].y);
        //-- 几何的点才有可能存在像素点关联，因而这里可视化这种像素点关联
        if(pts_geo[i].mbAssociated == true){
            int asso_edge_ID = pts_geo[i].asso_edge_ID;
            int asso_point_index = pts_geo[i].asso_point_index;
            int ref_idx = frame_cur->mmIndexMap[asso_edge_ID];
            orderedEdgePoint pt_asso = frame_cur->mvEdges[ref_idx].mvPoints[asso_point_index];
            cv::Point asso_pixel(pt_asso.x, pt_asso.y);
            cv::line(img_viz, geo_pixel, asso_pixel, cv::Scalar(160,255,255), 1, cv::LINE_AA);
            cv::circle(img_viz, asso_pixel, 1, cv::Scalar(0,255,0), -1, cv::LINE_AA);
        }
        cv::circle(img_viz, geo_pixel, 1, cv::Scalar(255,150,0), -1, cv::LINE_AA);
    }

    return img_viz;
}

bool checkPoseJump(Sophus::SE3d pose)
{
    Eigen::Matrix3d R = pose.rotationMatrix();
    Eigen::Vector3d t = pose.translation();
    double t_thres = 0.10;
    double angle_thres = 15.0;
    Eigen::AngleAxisd rotationVector(R);  
    Eigen::Vector3d axis = rotationVector.axis();  
    double angle = rotationVector.angle() * 180.0 / M_PI;
    if(t_thres < t.norm() || angle > angle_thres){
        return true;
    }else{
        return false;
    }
}

int main(int argc, char **argv) {

    //-- 读取序列配置文件
    std::string tum_name = argv[1];
    param::paramHandler ph("../config/" + tum_name + ".yaml");
    
    std::string tum_dir = ph.dataset_dir;
    
    std::string path_tum = tum_dir + tum_name + "/";
    std::vector<std::string> rgb_file_seq;
    std::vector<std::string> depth_file_seq;
    std::vector<double> rgb_stamp_seq;
    std::vector<double> depth_stamp_seq;
    tum_file::getTUMsequence(path_tum,rgb_file_seq,depth_file_seq,rgb_stamp_seq,depth_stamp_seq);

    std::cout<<"\033[32m[dataset] \033[0m"<<": finish loading file sequence"<<std::endl;

    std::vector<double> coarse_stamps;
    std::vector<Eigen::Matrix4d> coarse_poses;
    tum_file::getGTsequence(path_tum+"poses_coarse.txt", coarse_poses, coarse_stamps);

    fx = ph.fx;
    fy = ph.fy;
    cx = ph.cx;
    cy = ph.cy;
    geo_photo_ratio = ph.fine.geo_photo_ratio;
    kf_rot_thres = ph.fine.kf_rot_thres;
    kf_trans_thres = ph.fine.kf_trans_thres;
    sample_bias = ph.fine.sample_bias;

    int N = rgb_file_seq.size();
    float mDepthMapFactor = ph.depth_scale;

    //-- 上一帧（也就是上一个关键帧）的相关信息
    FramePtr kf_cur;
    Eigen::Matrix4d pose_last_coarse;

    fine::FineTracker tracker(fx, fy, cx, cy, geo_photo_ratio);

    cv::namedWindow("edges", cv::WINDOW_NORMAL);
    std::vector<Sophus::SE3d> poseVectorCurFrame; //-- 当前帧相对于自己的参考帧的位姿变换
    std::vector<int> cur2RefFrame;                //-- 当前帧相对的参考帧在参考帧列表的index
    std::vector<Sophus::SE3d> poseVectorRefFrame; //-- 参考帧的位姿变换
    std::cout<<"start fine tracking"<<std::endl;
    Eigen::Matrix4d NaN = Eigen::MatrixXd::Zero(4,4);
    int cnt = 0;
    for(int i = 0; i < N; ++i)
    {
        //-- 生成当前帧
        cv::Mat imgRGB = cv::imread(path_tum + rgb_file_seq[i], CV_LOAD_IMAGE_COLOR);
        cv::Mat imgDepth = cv::imread(path_tum + depth_file_seq[i], CV_LOAD_IMAGE_UNCHANGED);
        imgDepth.convertTo(imgDepth, CV_32F, mDepthMapFactor);
        
        edgeSelector selector(20.0, ph.fine.canny_low, ph.fine.canny_high);
        selector.processImage(imgRGB);

        //-- 得到当前帧的粗匹配位姿
        Eigen::Matrix4d pose_curr_coarse = tum_pose::getStaticGTPose(rgb_stamp_seq[i], coarse_stamps, coarse_poses);
        
        //-- 第一帧做参考帧
        if(cnt == 0){
            pose_last_coarse = pose_curr_coarse;
            cnt++;
            Sophus::SE3d identity;

            kf_cur.reset(new Frame(i, selector.mvEdges, imgRGB, imgDepth, fx, fy, cx, cy));
            tracker.setCurrent(kf_cur);

            poseVectorRefFrame.push_back(identity);
            poseVectorCurFrame.push_back(identity);
            cur2RefFrame.push_back(0);
            continue;
        }

        //-- 创建当前帧
        FramePtr kf_ref;
        kf_ref.reset(new Frame(i, selector.mvEdges, imgRGB, imgDepth, fx, fy, cx, cy));
        kf_ref->getFineSampledPoints(sample_bias);

        //-- 参考帧到当前帧的位姿变换
        Eigen::Matrix4d T_ref_cur_coarse = pose_last_coarse.inverse() * pose_curr_coarse;
        Sophus::SE3d pose_initial = convertTrans2Sophus(T_ref_cur_coarse);
        Sophus::SE3d pose_final;

        auto start_timer_asso = std::chrono::steady_clock::now();

        tracker.setReference(kf_ref);
        tracker.setPosePriorCur2Ref(pose_initial);
        tracker.estimate(pose_final, false);
        pose_final = pose_final.inverse();
        
        auto end_timer_asso = std::chrono::steady_clock::now();
        auto dt_asso = std::chrono::duration<double, std::milli>(end_timer_asso - start_timer_asso).count();
        std::cout<<"fine time:"<<dt_asso<<std::endl;

        if(checkPoseJump(pose_final)){
            std::cout<<"\033[33m [WARNING] \033[0m"<<rgb_stamp_seq[i]<<" : pose jumped, remain with coarse result!"<<std::endl;
            pose_final = pose_initial;
        }

        cv::Mat img_viz = visualizeReprojectionBias(tracker.getGeoPoints(), kf_cur, imgRGB);
        cv::imshow("edges", img_viz);
        if(cv::waitKey(1)==27)break;
        cnt++;
        
        
        poseVectorCurFrame.push_back(pose_final);
        cur2RefFrame.push_back(poseVectorRefFrame.size()-1);

        //-- 得到当前相对于关键帧的平移与旋转
        Eigen::Matrix4d trans = pose_final.matrix();
        Eigen::Matrix3d R = trans.block(0,0,3,3);
        Eigen::Vector3d t = trans.block(0,3,3,1);
        Eigen::AngleAxisd rotation_vector(R);  
        double theta = rotation_vector.angle() * 180 / M_PI; // 转换为角度 
        double translation = t.norm();

        if(theta > kf_rot_thres || translation > kf_trans_thres) 
        {
            //-- 用当前的图像创建新的关键帧
            kf_cur.reset(new Frame(i, selector.mvEdges, imgRGB, imgDepth, fx, fy, cx, cy));
            tracker.setCurrent(kf_cur);

            pose_last_coarse = pose_curr_coarse;
            
            //pose_gn是当前关键帧关于上一个关键帧的位姿，因此当前关键帧的全局位姿可以由上一个关键帧前推
            Sophus::SE3d frame_ref_global = poseVectorRefFrame.back() * pose_final;
            poseVectorRefFrame.push_back(frame_ref_global);
        }
    }
    cv::destroyAllWindows();

    //-- 获得从单位SE3为起点的位姿列表，时间戳为rgb_stamp_seq,
    std::vector<Eigen::Matrix4d> poseGlobal;
    pcl::PointCloud<pcl::PointXYZI>::Ptr traj(new pcl::PointCloud<pcl::PointXYZI>);
    for(int i = 0; i < poseVectorCurFrame.size(); ++i){
        int refIndex = cur2RefFrame[i];
        Sophus::SE3d poseRef = poseVectorRefFrame[refIndex];
        Sophus::SE3d poseCur = poseVectorCurFrame[i];
        Sophus::SE3d poseG = poseRef * poseCur;
        Eigen::Matrix4d trans_global = poseG.matrix();
        poseGlobal.push_back(trans_global);
    }

    //-- 根据rgb_stamp_seq的时间戳寻找对应的真值位姿，并将上述的位姿列表修改为与gt共起点
    std::vector<Eigen::Matrix4d> poseList;
    std::vector<double> stamps_final;
    Eigen::Matrix4d initialBias = Eigen::MatrixXd::Identity(4,4);
    // initialBias = tum_pose::getStaticGTPose(rgb_stamp_seq[0], gt_stamps, gt_poses);//-- 第一帧位姿（原始位姿对应E）的零漂
    // std::cout<<initialBias<<std::endl;
    bool isFirstValid = false;
    for(int i = 0; i < N; ++i){
        // //-- 如果这是第一帧有效得到真值的，则以这一帧作为基准
        if(isFirstValid == false){
            isFirstValid = true;
        }
        Eigen::Matrix4d pose_curr = poseGlobal[i];
        // //-- 检查是否有不合理的位姿
        // if(pose_curr(0,3)>100 || pose_curr(1,3)>100 || pose_curr(2,3)>100){
        //     continue;
        // }
        pose_curr = initialBias * pose_curr;
        poseList.push_back(pose_curr);
        stamps_final.push_back(rgb_stamp_seq[i]);
        pcl::PointXYZI pt;
        pt.x = pose_curr(0,3);
        pt.y = pose_curr(1,3);
        pt.z = pose_curr(2,3);
        pt.intensity = poseList.size();
        traj->push_back(pt);
    }
    std::cout<<traj->points.size()<<std::endl;
    std::string storeFileName = path_tum + "poses_fine.txt";
    tum_file::saveEvaluationFiles(storeFileName, poseList, stamps_final);



    //-- 可视化一个强度点云
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (1, 1, 1, 0);
    viewer->addCoordinateSystem(0.5);
    pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> ints(traj, "intensity");
    viewer->addPointCloud<pcl::PointXYZI> (traj, ints, "undistortP");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "undistortP");
    while (!viewer->wasStopped()) {
        viewer->spinOnce();
    }

    return 0;
}
