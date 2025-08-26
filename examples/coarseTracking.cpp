#include <iostream>
#include <fstream>
#include <sophus/se3.hpp>

#include <chrono> //-- 计时函数

#include <pcl/console/parse.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transforms.h>

#include "edgeSelector.h"
#include "Frame.h"
#include "directTracker.h"
#include "tum_pose.h"
#include "tum_file.h"

#include "paramLoader.h"

//-- TUM的相机参数
float fx; 
float fy; 
float cx; 
float cy;
int sample_bias;
int maximum_point;
double kf_rot_thres;
double kf_trans_thres;

double degreesToPiRange(double degrees) {
    assert(degrees >=0 && degrees < 360.0);

    // 映射到 [-π, π]
    if (degrees <= 180) {
        return degrees * M_PI / 180.0; // 0~180 → 0~π
    } else {
        return (degrees - 360) * M_PI / 180.0; // 180~360 → -π~0
    }
}

void generateSrcPixelsSampled(FramePtr frame_cur, std::vector<float>& edge_point_total_x, 
                       std::vector<float>& edge_point_total_y, std::vector<float>& edge_depth_total,
                       std::vector<float>& edge_weight_total, std::vector<cv::Point3f>& cloud_total,
                       std::vector<float>& edge_point_total_theta)
{
    edge_point_total_x.clear(); edge_point_total_y.clear();
    edge_depth_total.clear(); edge_weight_total.clear();
    cloud_total.clear();
    edge_point_total_theta.clear();
    
    std::vector<orderedEdgePoint> sampledPoints = frame_cur->getCoarseSampledPoints(sample_bias, maximum_point);
    for(int i = 0; i < sampledPoints.size(); ++i){
        orderedEdgePoint pt = sampledPoints[i];
        float angle = pt.imgGradAngle;
        angle = degreesToPiRange(angle);
        float depth = pt.depth;
        float weight = pt.score_depth;
        cv::Point3f cloud_pt = cv::Point3f(pt.x_3d, pt.y_3d, pt.z_3d);
        edge_depth_total.push_back(depth);
        edge_weight_total.push_back(weight);
        edge_point_total_x.push_back(pt.x);
        edge_point_total_y.push_back(pt.y);
        cloud_total.push_back(cloud_pt);
        edge_point_total_theta.push_back(angle);
    }
}


cv::Mat visualizeEstimationResult(const std::vector<cv::Point3f>& points_3d_orig,
                               Sophus::SE3d& pose,
                               cv::Mat img_BackGround_orig,
                               float fx, float fy, float cx, float cy)
{
    cv::Mat img_BackGround = img_BackGround_orig.clone();
    std::vector<cv::Point> pixel_orig;
    std::vector<cv::Point> pixel_trans;
    for(int i = 0; i < points_3d_orig.size(); ++i){
            //-- 使用当前位姿进行3D-2D投影，得到投影像素
            Eigen::Vector3d pt_3d = Eigen::Vector3d(points_3d_orig[i].x,
                                                    points_3d_orig[i].y,
                                                    points_3d_orig[i].z);
            Eigen::Vector3d pc_trans = pose * pt_3d;
            cv::Point proj(fx * pc_trans[0] / pc_trans[2] + cx, 
                           fy * pc_trans[1] / pc_trans[2] + cy);
            cv::Point orig(fx * pt_3d[0] / pt_3d[2] + cx, 
                           fy * pt_3d[1] / pt_3d[2] + cy);
            pixel_orig.push_back(orig);
            pixel_trans.push_back(proj);
    }
    if(img_BackGround.channels()==1){
        cv::cvtColor(img_BackGround, img_BackGround, cv::COLOR_GRAY2BGR);
    }
    for(int i = 0; i < pixel_orig.size(); ++i){
        cv::circle(img_BackGround, pixel_trans[i], 2, cv::Vec3b(0,255,0),-1, cv::LINE_AA);
        cv::circle(img_BackGround, pixel_orig[i], 2, cv::Vec3b(0,0,255),-1, cv::LINE_AA);
    }
    return img_BackGround;
}

double checkReprojectionBias(const std::vector<cv::Point3f>& points_3d_orig,
                             Sophus::SE3d& pose,
                             float fx, float fy, float cx, float cy)
{
    std::vector<cv::Point> pixel_orig;
    std::vector<cv::Point> pixel_trans;
    std::vector<double> pixel_bias;
    for(int i = 0; i < points_3d_orig.size(); ++i){
        //-- 使用当前位姿进行3D-2D投影，得到投影像素
        Eigen::Vector3d pt_3d = Eigen::Vector3d(points_3d_orig[i].x,
                                                points_3d_orig[i].y,
                                                points_3d_orig[i].z);
        Eigen::Vector3d pc_trans = pose * pt_3d;
        cv::Point proj(fx * pc_trans[0] / pc_trans[2] + cx, 
                        fy * pc_trans[1] / pc_trans[2] + cy);
        cv::Point orig(fx * pt_3d[0] / pt_3d[2] + cx, 
                        fy * pt_3d[1] / pt_3d[2] + cy);
        pixel_orig.push_back(orig);
        pixel_trans.push_back(proj);
        cv::Point bias = cv::Point(proj.x-orig.x, proj.y-orig.y);
        double bias_norm = sqrt(bias.x * bias.x + bias.y * bias.y);
        pixel_bias.push_back(bias_norm);
    }
    //-- 计算重投影的像素点对的差异
    //-- 计算偏差的平均值
    double sum = 0.0;
    double meanVal = 0.0;
    for (const auto& value : pixel_bias){
        sum += value;
    }
    meanVal = sum / static_cast<double>(pixel_bias.size());
    //-- 计算偏差的中位值
    std::size_t size = pixel_bias.size();
    std::sort(pixel_bias.begin(), pixel_bias.end());
    if (size % 2 == 0){
        return (pixel_bias[size / 2 - 1] + pixel_bias[size / 2]) / 2.0;
    }else{
        return pixel_bias[size / 2];
    }

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
    std::cout<<"\033[32m[dataset] \033[0m"<<": finish loading file sequence\n";

    int N = rgb_file_seq.size();

    float scale = ph.depth_scale;
    fx = ph.fx;
    fy = ph.fy;
    cx = ph.cx;
    cy = ph.cy;
    sample_bias = ph.coarse.sample_bias;
    maximum_point = ph.coarse.maximum_point;
    kf_rot_thres = ph.coarse.kf_rot_thres;
    kf_trans_thres = ph.coarse.kf_trans_thres;
    int width = ph.coarse.width;
    int height = ph.coarse.height;

    FramePtr pFrameRef;             //-- 参考关键帧的智能指针
    cv::Mat refMatGray;             //-- 参考关键帧的灰度图
    bool isKeyFrameChanged = false; //-- 用于在循环中检查有没有换过关键帧

    //-- 用于构造直接法参考帧的边缘点信息
    std::vector<float> x_list, y_list, depth_list, weight_list, theta_list;
    std::vector<cv::Point3f> frame_cloud_ref;

    cv::namedWindow("edges", cv::WINDOW_NORMAL);

    std::vector<double> listTimeStamp;             //-- 位姿轨迹对应的时间戳
    std::vector<Sophus::SE3d> poseVectorCurFrame; //-- 当前帧相对于自己的参考帧的位姿变换
    std::vector<int> cur2RefFrame;                //-- 当前帧相对的参考帧在参考帧列表的index
    std::vector<Sophus::SE3d> poseVectorRefFrame; //-- 参考帧的位姿变换

    edgeSelector selector(20.0, ph.coarse.canny_low, ph.coarse.canny_high);

    direct::DirectTracker coarse_tracker(width, height, ph.fx, ph.fy, ph.cx, ph.cy);

    for(int i = 0; i < N; ++i)
    {
        cv::Mat imgRGB = cv::imread(path_tum + rgb_file_seq[i], CV_LOAD_IMAGE_COLOR);
        cv::Mat imgGray;
        cv::cvtColor(imgRGB, imgGray, cv::COLOR_BGR2GRAY);
        cv::Mat imgDepth = cv::imread(path_tum + depth_file_seq[i], CV_LOAD_IMAGE_UNCHANGED);
        imgDepth.convertTo(imgDepth, CV_32F, scale);
        
        if(i == 0)
        {
            //-- 一开始的时候即创建一个参考关键帧
            auto start_timer = std::chrono::steady_clock::now();
            selector.processImage(imgRGB);
            auto end_timer = std::chrono::steady_clock::now();
            double dt = std::chrono::duration<double, std::milli>(end_timer - start_timer).count();
            std::cout<<"extraction time:"<<dt<<std::endl;
            pFrameRef.reset(new Frame(i, selector.mvEdges, imgRGB, imgDepth, ph.fx, ph.fy, ph.cx, ph.cy));
            refMatGray = imgGray.clone();
            isKeyFrameChanged = true;
            
            //-- 首帧位姿为 Identity
            Sophus::SE3d identity;
            poseVectorRefFrame.push_back(identity);
            poseVectorCurFrame.push_back(identity);
            listTimeStamp.push_back(rgb_stamp_seq[i]);
            cur2RefFrame.push_back(0);
            continue;
        }
        
        bool isDataValid = false;
        if(isKeyFrameChanged)
        {
            generateSrcPixelsSampled(pFrameRef, x_list, y_list, depth_list, weight_list, frame_cloud_ref, theta_list);
            //-- 重置粗匹配的参考帧
            coarse_tracker.setReference(refMatGray, x_list, y_list, depth_list, weight_list, theta_list);
            isKeyFrameChanged = false;
        }

        if(x_list.size()>200) isDataValid = true;
        
        Sophus::SE3d pose_gn;

        // if(poseVectorCurFrame.size() > 0)
        // {
        //     pose_gn = poseVectorCurFrame.back();
        // }

        if(isDataValid){
            auto start_timer = std::chrono::steady_clock::now();
            //-- 设置当前帧
            coarse_tracker.setCurrent(imgGray);
            //-- 并行计算当前帧到参考帧的位姿变换
            coarse_tracker.estimatePyramid(pose_gn, true);
            auto end_timer = std::chrono::steady_clock::now();
            float dt = std::chrono::duration<double, std::milli>(end_timer - start_timer).count();
            std::cout<<"coarse time:"<<dt<<std::endl;
        }

        // if(checkPoseJump(pose_gn)){
        if(checkPoseJump(pose_gn) || isDataValid == false){
            std::cout<<"FAILED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"<<std::endl;
            std::cout<<std::setprecision(20)<<rgb_stamp_seq[i]<<std::endl;
        }
        
        //Sophus::SE3d pose_gn = pose_gn_inverse.inverse();
        cv::Mat imgViz = visualizeEstimationResult(frame_cloud_ref, pose_gn, imgRGB, ph.fx, ph.fy, ph.cx, ph.cy);
        cv::imshow("edges", imgViz);
        cv::waitKey(1);

        //-- 参考帧到当前帧的位姿变换应该是直接法推出的结果的求逆
        pose_gn = pose_gn.inverse();


        poseVectorCurFrame.push_back(pose_gn);
        listTimeStamp.push_back(rgb_stamp_seq[i]);
        cur2RefFrame.push_back(poseVectorRefFrame.size()-1);
        //-- 得到当前相对于关键帧的平移与旋转
        Eigen::Matrix4d trans = pose_gn.matrix();
        Eigen::Matrix3d R = trans.block(0,0,3,3);
        Eigen::Vector3d t = trans.block(0,3,3,1);
        Eigen::AngleAxisd rotation_vector(R);  
        double theta = rotation_vector.angle() * 180 / M_PI; // 转换为角度 
        double translation = t.norm();
        
        
        bool influential = false;
        //-- 通过平移和旋转来界定关键帧
        if(theta > kf_rot_thres|| translation > kf_trans_thres) influential = true; //-- 4 0.03 is the best
        //-- 通过重投影偏差来界定关键帧
        //if(checkReprojectionBias(frame_cloud_ref, pose_gn, fx, fy, cx, cy) > 15) influential = true;
        if( influential == true)
        {
            //-- 一开始的时候即创建一个参考关键帧
            auto start_timer = std::chrono::steady_clock::now();
            selector.processImage(imgRGB);
            auto end_timer = std::chrono::steady_clock::now();
            double dt = std::chrono::duration<double, std::milli>(end_timer - start_timer).count();
            std::cout<<"extraction time:"<<dt<<std::endl;
            pFrameRef.reset(new Frame(i, selector.mvEdges, imgRGB, imgDepth, ph.fx, ph.fy, ph.cx, ph.cy));
            refMatGray = imgGray.clone();
            isKeyFrameChanged = true;

            //pose_gn是当前关键帧关于上一个关键帧的位姿，因此当前关键帧的全局位姿可以由上一个关键帧前推
            Sophus::SE3d frame_ref_global = poseVectorRefFrame.back() * pose_gn;
            poseVectorRefFrame.push_back(frame_ref_global);
        }
        
    }
    cv::destroyAllWindows();

    std::cout<<"\033[34m"<<"[tracking]"<<"\033[0m"<<" : finish tracking"<<std::endl;

    //-- 获得从单位SE3为起点的位姿列表，时间戳为rgb_stamp_seq,
    std::vector<Eigen::Matrix4d> poseGlobal;
    pcl::PointCloud<pcl::PointXYZI>::Ptr traj(new pcl::PointCloud<pcl::PointXYZI>);
    for(int i = 0; i < poseVectorCurFrame.size(); ++i)
    {
        int refIndex = cur2RefFrame[i];
        Sophus::SE3d poseRef = poseVectorRefFrame[refIndex];
        Sophus::SE3d poseCur = poseVectorCurFrame[i];
        Sophus::SE3d poseG = poseRef * poseCur;
        Eigen::Matrix4d trans_global = poseG.matrix();
        poseGlobal.push_back(trans_global);

        pcl::PointXYZI pt;
        pt.x = trans_global(0,3);
        pt.y = trans_global(1,3);
        pt.z = trans_global(2,3);
        pt.intensity = poseGlobal.size();
        traj->push_back(pt);
    }

    std::cout<<traj->points.size()<<std::endl;
    std::string storeFileName = path_tum + "poses_coarse.txt";
    tum_file::saveEvaluationFiles(storeFileName, poseGlobal, listTimeStamp);




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
