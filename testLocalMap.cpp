/**
 * @brief 示例例程，读取ICL数据集的rgb图与rgbd图并进行深度信息融合
 * @details 读取RGBD数据集的数据，图像数据在rgb文件夹下，深度数据在depth文件夹下
 * @author 猎魔人
 * @date 2024-5-29
 * @details 参考ORB-SLAM2中读取RGBD序列的方法
*/
#include <iostream>
#include <fstream>
#include <sophus/se3.hpp>

#include "edgeSelector.h"
#include "Frame.h"
#include "tum_pose.h"
#include "tum_file.h"
#include "KeyFrame.h"
#include "localMap.h"
#include "paramLoader.h"
#include "visualizerPCL.h"

//-- TUM的相机参数
float fx; 
float fy; 
float cx; 
float cy;

//-- 得到以局部地图的基准坐标系为零坐标系的局部地图点云
void visualizeAssociationResult(const edge_map::localMap& l_map,
                                std::vector<pcl::PointCloud<pcl::PointXYZ>>& clusterClouds)
{
    clusterClouds.clear();
    for(const auto& pair : l_map.mmElementEdgeMap)
    {
        std::vector<int> edgeIdx = pair.second;

        //-- 整个边缘地图元素聚类的点云
        pcl::PointCloud<pcl::PointXYZ> clusterCloud;

        for(int i = 0; i < edgeIdx.size(); ++i){
            int kf_edge_idx = l_map.mvElementEdges[edgeIdx[i]].kf_edge_idx;
            int kf_id = l_map.mvElementEdges[edgeIdx[i]].kf_id;
            int kf_idx = l_map.mmKFID2KFindex.at(kf_id);

            Edge& edge = l_map.mvKeyFrames[kf_idx]->mvEdges[kf_edge_idx];
            //-- 当前地图元素边缘的全局位姿
            Eigen::Matrix4d Trans_curr = l_map.mvKeyFrames[kf_idx]->KF_pose_g.matrix();
            //-- 当前地图元素边缘相对于局部地图基准坐标系的相对位姿
            Eigen::Matrix4d Trans_ref_curr = l_map.T_ref.inverse().matrix() * Trans_curr;
            //-- 单个边缘地图元素的点云
            pcl::PointCloud<pcl::PointXYZ> cloud;
            //-- 计算单个边缘地图元素的点云
            for(int j = 0; j < edge.mvPoints.size(); ++j){
                orderedEdgePoint pt = edge.mvPoints[j];
                //-- 计算3D坐标
                float x = (float(pt.x) - cx)/fx * pt.depth;
                float y = (float(pt.y) - cy)/fy * pt.depth;
                float z = pt.depth;
                Eigen::Vector4d points(x,y,z,1);
                //-- 重投影得到新的投影点
                points = Trans_ref_curr*points;
                pcl::PointXYZ point(points.x(),points.y(),points.z());
                cloud.points.push_back(point);
            }
            clusterCloud += cloud;
        }
        clusterClouds.push_back(clusterCloud);
    }
}

int main(int argc, char **argv) {
    std::string tum_dir = "/home/lab/slamData/TUM-RGBD/";
    std::string tum_name = argv[1];
    std::string path_tum = tum_dir + tum_name + "/";
    std::vector<std::string> rgb_file_seq;
    std::vector<std::string> depth_file_seq;
    std::vector<double> rgb_stamp_seq;
    std::vector<double> depth_stamp_seq;
    tum_file::getTUMsequence(path_tum,rgb_file_seq,depth_file_seq,rgb_stamp_seq,depth_stamp_seq);

    std::vector<double> coarse_stamps;
    std::vector<Eigen::Matrix4d> coarse_poses;
    tum_file::getGTsequence(path_tum+"poses_coarse.txt", coarse_poses, coarse_stamps);

    param::paramHandler ph("../config/" + tum_name + ".yaml");
    fx = ph.fx;
    fy = ph.fy;
    cx = ph.cx;
    cy = ph.cy;

    int N = rgb_file_seq.size();
    float mDepthMapFactor = ph.depth_scale;

    std::cout<<"start association"<<std::endl;
    int cnt = 0;
    std::vector<KeyFramePtr> pKF_list;
    edge_map::localMap l_map;
    Eigen::Matrix4d pose_last;
    for(int i = 0; i < 50; ++i){
        //-- 得到当前帧的粗匹配位姿
        Eigen::Matrix4d pose_curr_coarse = tum_pose::getStaticGTPose(rgb_stamp_seq[i], coarse_stamps, coarse_poses);
        Eigen::Matrix3d R = pose_curr_coarse.block<3, 3>(0, 0);
        Eigen::Vector3d t = pose_curr_coarse.block<3, 1>(0, 3);
        // 创建 SE3 对象
        // 注意: Sophus::SE3d 构造函数需要先传入旋转，再传入平移
        Sophus::SE3d T(R, t);

        if(i == 0)
        {
            pose_last = pose_curr_coarse;
        }

        //-- 得到当前相对于关键帧的平移与旋转
        Eigen::Matrix4d trans = pose_last.inverse() * pose_curr_coarse;
        Eigen::Matrix3d R_bias = trans.block(0,0,3,3);
        Eigen::Vector3d t_bias = trans.block(0,3,3,1);
        Eigen::AngleAxisd rotation_vector(R_bias);  
        double theta = rotation_vector.angle() * 180 / M_PI; // 转换为角度 
        double translation = t_bias.norm();

        bool select = theta > std::atof(argv[2]) || translation > std::atof(argv[3]);
        if (select || i == 0)
        {
            //-- 生成当前帧
            cv::Mat imgRGB = cv::imread(path_tum + rgb_file_seq[i], CV_LOAD_IMAGE_COLOR);
            cv::Mat imgDepth = cv::imread(path_tum + depth_file_seq[i], CV_LOAD_IMAGE_UNCHANGED);
            imgDepth.convertTo(imgDepth, CV_32F, mDepthMapFactor);
            edgeSelector selector(20.0, 40, 80);
            selector.processImage(imgRGB);
            Frame frame_cur(i, selector.mvEdges, imgRGB, imgDepth, ph.fx, ph.fy, ph.cx, ph.cy);
            frame_cur.getFineSampledPoints(ph.fine.sample_bias);

            //-- 根据帧信息与位姿信息创建一个关键帧
            KeyFramePtr pKF( new KeyFrame(i, T, rgb_stamp_seq[i], selector.mvEdges, imgRGB, imgDepth, ph.fx, ph.fy, ph.cx, ph.cy));
            pKF->getFineSampledPoints(ph.fine.sample_bias);
            

            l_map.mvKeyFrames.push_back(pKF);
            l_map.mmKFID2KFindex[pKF->KF_ID] = l_map.mvKeyFrames.size()-1;
        }


       if (l_map.mvKeyFrames.size()>=2) break;
    }

    auto start_timer_asso = std::chrono::steady_clock::now();

    l_map.initLocalMap();
    l_map.generateLabelMap();

    auto end_timer_asso = std::chrono::steady_clock::now();
    auto dt_asso = std::chrono::duration<double, std::milli>(end_timer_asso - start_timer_asso).count();
    std::cout<<"fine time:"<<dt_asso<<std::endl;

    std::vector<pcl::PointCloud<pcl::PointXYZ>> clusterClouds;
    visualizeAssociationResult(l_map, clusterClouds);

    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
	viewer->setBackgroundColor(1,1,1);
    // viewer->addCoordinateSystem(0.5);

    //-- 绘制局部地图的聚类点云
    for(int i = 0; i < clusterClouds.size(); ++i){
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
        *cloud_cluster = clusterClouds[i];
        std::string id = "cloud_cluster_gt_" + std::to_string(i);
        // Generate a random (bright) color
        viz_util::drawPointCloudRandomColor(viewer, id, cloud_cluster, 3);
    }

    //-- 绘制参与局部地图的所有关键帧的相机
    std::vector<Eigen::Matrix4d> pose_list;
    for(int i = 0; i < l_map.mvKeyFrames.size(); ++i)
    {
        Sophus::SE3d pose = l_map.mvKeyFrames[i]->KF_pose_g;
        Sophus::SE3d pose_ref_cur = l_map.T_ref.inverse() * l_map.mvKeyFrames[i]->KF_pose_g;
        viz_util::drawCamera(viewer, pose_ref_cur.matrix(), "camera"+std::to_string(i), 0.08, 0, 100, 255, 2);
    }

    while (!viewer->wasStopped()) {
        viewer->spinOnce();
    }

    return 0;
}
