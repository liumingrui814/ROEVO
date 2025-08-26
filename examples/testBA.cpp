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
#include "Optimizer.h"

//-- TUM的相机参数
float fx; 
float fy; 
float cx; 
float cy;

//-- 得到以局部地图的基准坐标系为零坐标系的局部地图点云
void visualizeAssociationResult(const edge_map::localMapPtr& pLocalMap,
                                std::vector<pcl::PointCloud<pcl::PointXYZ>>& clusterClouds,
                                std::vector<cv::Vec3b>& clusterCloudColors)
{
    clusterClouds.clear();
    for(const auto& cluster : pLocalMap->mvEleEdgeClusters)
    {
        std::vector<unsigned int> edgeIDs = cluster.mvElementEdgeIDs;
        std::vector<int> edgeIdx;
        if(edgeIDs.size() < 1) continue;
        for(int i = 0; i < edgeIDs.size(); ++i)
        {
            int index = pLocalMap->mmElementID2index.at(edgeIDs[i]);
            edgeIdx.push_back(index);
        }

        //-- 整个边缘地图元素聚类的点云
        pcl::PointCloud<pcl::PointXYZ> clusterCloud;
        cv::Vec3b color = cluster.visColor;

        for(int i = 0; i < edgeIdx.size(); ++i){
            int kf_edge_idx = pLocalMap->mvElementEdges[edgeIdx[i]].kf_edge_idx;
            int kf_id = pLocalMap->mvElementEdges[edgeIdx[i]].kf_id;
            int kf_idx = pLocalMap->mmKFID2KFindex.at(kf_id);

            Edge& edge = pLocalMap->mvKeyFrames[kf_idx]->mvEdges[kf_edge_idx];
            //-- 当前地图元素边缘的全局位姿
            Eigen::Matrix4d Trans_curr = pLocalMap->mvKeyFrames[kf_idx]->KF_pose_g.matrix();
            //-- 当前地图元素边缘相对于局部地图基准坐标系的相对位姿
            Eigen::Matrix4d Trans_ref_curr = Trans_curr;
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
        clusterCloudColors.push_back(color);
    }
}


void visualizeMergedLocalMap(const edge_map::localMapPtr& pLocalMap,
                             std::vector<pcl::PointCloud<pcl::PointXYZ>>& mergedClouds)
{
    mergedClouds.clear();
    mergedClouds.reserve(pLocalMap->mvEleEdgeClusters.size());

    for(const auto& cluster : pLocalMap->mvEleEdgeClusters)
    {
        std::vector<cv::Point3d> merged_cloud = cluster.mvMergedCloud_ref;
        size_t N = merged_cloud.size();
        pcl::PointCloud<pcl::PointXYZ> cloud;
        cloud.points.reserve(N);
        for(size_t i = 0; i < N; ++i)
        {
            Eigen::Vector3d pt(merged_cloud[i].x, merged_cloud[i].y, merged_cloud[i].z);
            //-- 投影到世界坐标系下
            //pt = pose_ref * pt;
            cloud.points.emplace_back(pt.x(),pt.y(),pt.z());
        }
        mergedClouds.push_back(cloud);
    }
}

void visualizeAssociatedCloud3D(std::vector<match3d_2d>& matches, KeyFramePtr pKF, pcl::PointCloud<pcl::PointXYZ>& cloud)
{
    std::vector<Edge> edges;
    for(size_t i = 0; i < matches.size(); ++i)
    {
        const match3d_2d& match = matches[i];
        std::vector<edge_map::elementEdge> ele_edges = match.second;
        for(size_t j = 0; j < ele_edges.size(); ++j)
        {
            edge_map::elementEdge ele_edge = ele_edges[j];
            int index = ele_edge.kf_edge_idx;
            Edge& edge = pKF->mvEdges.at(index);
            edges.push_back(edge);
        }
    }

    pcl::PointCloud<pcl::PointXYZ> cloud_kf;
    Sophus::SE3d pose_global = pKF->KF_pose_g;
    for(size_t i = 0; i < edges.size(); ++i)
    {
        Edge& edge = edges[i];
        for(size_t j = 0; j < edge.mvPoints.size(); ++j)
        {
            const orderedEdgePoint& pt = edge.mvPoints[j];
            Eigen::Vector3d point(pt.x_3d, pt.y_3d, pt.z_3d);
            Eigen::Vector3d point_global = pose_global * point;
            cloud_kf.push_back(pcl::PointXYZ(point_global.x(),point_global.y(), point_global.z()));
        }
    }
    cloud = std::move(cloud_kf);
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
    tum_file::getGTsequence(path_tum+"poses_fine.txt", coarse_poses, coarse_stamps);

    fx = ph.fx;
    fy = ph.fy;
    cx = ph.cx;
    cy = ph.cy;

    int N = rgb_file_seq.size();
    float mDepthMapFactor = ph.depth_scale;

    std::cout<<"start association"<<std::endl;
    int cnt = 0;
    std::vector<KeyFramePtr> pKF_list;
    edge_map::localMapPtr pLocalMap(new edge_map::localMap());
    Eigen::Matrix4d pose_last;
    bool is_initialized = false;

    edgeSelector selector(20.0, 30, 50);

    for(int i = 0; i < N; ++i){
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

        bool select = theta > ph.win.kf_rot_thres || translation > ph.win.kf_trans_thres;
        if (select || i == 0)
        {
            pose_last = pose_curr_coarse;

            //-- 生成当前帧
            cv::Mat imgRGB = cv::imread(path_tum + rgb_file_seq[i], CV_LOAD_IMAGE_COLOR);
            cv::Mat imgDepth = cv::imread(path_tum + depth_file_seq[i], CV_LOAD_IMAGE_UNCHANGED);
            imgDepth.convertTo(imgDepth, CV_32F, mDepthMapFactor);
            selector.processImage(imgRGB);

            //-- 根据帧信息与位姿信息创建一个关键帧
            KeyFramePtr pKF( new KeyFrame(i, T, rgb_stamp_seq[i], selector.mvEdges, imgRGB, imgDepth, ph.fx, ph.fy, ph.cx, ph.cy));
            pKF->getFineSampledPoints(ph.fine.sample_bias);

            std::cout<<"add frame"<<std::endl;
            pLocalMap->addFrame2LocalMap(pKF);

            if(pLocalMap->mvKeyFrames.size() == 10)
                break;
        }
    }

    std::vector<Sophus::SE3d> poses_old_camera;
    for(int i = 0; i < pLocalMap->mvKeyFrames.size(); ++i)
    {
        Sophus::SE3d pose = pLocalMap->mvKeyFrames[i]->KF_pose_g;
        poses_old_camera.push_back(pose);
    }

    std::vector<double> list_kf_stamps_orig;
    std::vector<Eigen::Matrix4d> list_kf_poses_orig;
    for(size_t i = 0; i < pLocalMap->mvKeyFrames.size(); ++i)
    {
        double stamp = pLocalMap->mvKeyFrames[i]->KF_stamp;
        Eigen::Matrix4d pose = pLocalMap->mvKeyFrames[i]->KF_pose_g.matrix();
        list_kf_poses_orig.push_back(pose);
        list_kf_stamps_orig.push_back(stamp);
    }
    std::string storeFileName = path_tum + "posesKF_old.txt";
    tum_file::saveEvaluationFiles(storeFileName, list_kf_poses_orig, list_kf_stamps_orig);

    std::cout<<"finish adding"<<std::endl;

    for(size_t i = 0; i < 1; ++i)
    {
        pLocalMap->clusterFittingProjection();
        edge_map::Optimizer::optimizeAllInvolvedKFs(pLocalMap);
    }

    std::cout<<"finish correlating"<<std::endl;

    std::vector<double> list_kf_stamps;
    std::vector<Eigen::Matrix4d> list_kf_poses;
    for(size_t i = 0; i < pLocalMap->mvKeyFrames.size(); ++i)
    {
        double stamp = pLocalMap->mvKeyFrames[i]->KF_stamp;
        Eigen::Matrix4d pose = pLocalMap->mvKeyFrames[i]->KF_pose_g.matrix();
        list_kf_poses.push_back(pose);
        list_kf_stamps.push_back(stamp);
    }
    storeFileName = path_tum + "posesKF.txt";
    tum_file::saveEvaluationFiles(storeFileName, list_kf_poses, list_kf_stamps);


    std::vector<pcl::PointCloud<pcl::PointXYZ>> mergedCloud;
    visualizeMergedLocalMap(pLocalMap, mergedCloud);

    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
	viewer->setBackgroundColor(1,1,1);
    // viewer->addCoordinateSystem(0.5);

    std::vector<pcl::PointCloud<pcl::PointXYZ>> clusterClouds;
    std::vector<cv::Vec3b> clusterCloudColors;
    visualizeAssociationResult(pLocalMap, clusterClouds, clusterCloudColors);

    //-- 绘制局部地图的聚类点云
    for(int i = 0; i < clusterClouds.size(); ++i){
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
        *cloud_cluster = clusterClouds[i];
        std::string id = "cloud_cluster_gt_" + std::to_string(i);
        cv::Vec3b color = clusterCloudColors[i];
        // Generate a random (bright) color
        viz_util::drawPointCloudColor(viewer, id, cloud_cluster, color[0], color[1], color[2], 1);
    }

    // //-- 绘制局部地图的融合边缘
    // for(int i = 0; i < mergedCloud.size(); ++i){
    //     pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
    //     *cloud_cluster = mergedCloud[i];
    //     std::string id = "cloud_cluster_mg_" + std::to_string(i);
    //     // Generate a random (bright) color
    //     viz_util::drawPointCloudColor(viewer, id, cloud_cluster, 255, 0, 0, 4);
    //     for(int j = 1; j < cloud_cluster->points.size(); ++j)
    //     {
    //         pcl::PointXYZ pt_1 = cloud_cluster->points[j-1];
    //         pcl::PointXYZ pt_2 = cloud_cluster->points[j];
    //         std::string id_line = id + "_line_" + std::to_string(j);
    //         viz_util::drawLine(viewer, id_line, pt_1, pt_2, 128, 128, 128, 1);
    //     }
    // }

    // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_kf_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    // *cloud_kf_ptr = cloud_kf;
    // viz_util::drawPointCloudColor(viewer, "ref_cloud", cloud_kf_ptr, 0, 200, 0, 3);

    //-- 绘制参与局部地图的所有关键帧的相机
    std::vector<Eigen::Matrix4d> pose_list;
    for(int i = 0; i < pLocalMap->mvKeyFrames.size(); ++i)
    {
        Sophus::SE3d pose = pLocalMap->mvKeyFrames[i]->KF_pose_g;
        Sophus::SE3d pose_ref_cur = pLocalMap->mvKeyFrames[i]->KF_pose_g;
        viz_util::drawCamera(viewer, pose_ref_cur.matrix(), "camera"+std::to_string(i), 0.08, 255, 100, 0, 2);
    }

    for(int i = 0; i < poses_old_camera.size(); ++i)
    {
        viz_util::drawCamera(viewer, poses_old_camera[i].matrix(), "camera_old_"+std::to_string(i), 0.08, 0, 100, 255, 2);
    }

    while (!viewer->wasStopped()) {
        viewer->spinOnce();
    }

    return 0;
}
