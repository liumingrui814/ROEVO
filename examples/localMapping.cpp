#include <iostream>
#include <fstream>
#include <sophus/se3.hpp>

#include <chrono>
#include <thread>

#include "edgeSelector.h"
#include "Frame.h"
#include "tum_pose.h"
#include "tum_file.h"
#include "KeyFrame.h"
#include "localMap.h"
#include "paramLoader.h"
#include "visualizerPCL.h"
#include "Optimizer.h"

#include "voViewer.h"

//-- TUM的相机参数
float fx; 
float fy; 
float cx; 
float cy;

//-- 得到以局部地图的基准坐标系为零坐标系的局部地图点云
void visualizeAssociationResult(const edge_map::localMapPtr& pLocalMap,
                                std::vector<std::vector<cv::Point3d>>& clusterClouds,
                                std::vector<cv::Vec3b>& clusterCloudColors)
{
    clusterClouds.clear();
    clusterCloudColors.clear();
    for(const auto& cluster : pLocalMap->mvEleEdgeClusters)
    {
        std::vector<unsigned int> edgeIDs = cluster.mvElementEdgeIDs;
        std::vector<int> edgeIdx;
        if(edgeIDs.size() < 5) continue;
        for(int i = 0; i < edgeIDs.size(); ++i)
        {
            int index = pLocalMap->mmElementID2index.at(edgeIDs[i]);
            edgeIdx.push_back(index);
        }

        //-- 整个边缘地图元素聚类的点云
        std::vector<cv::Point3d> clusterCloud;
        cv::Vec3b color = cluster.visColor;

        for(int i = 0; i < edgeIdx.size(); ++i){
            int kf_edge_idx = pLocalMap->mvElementEdges[edgeIdx[i]].kf_edge_idx;
            int kf_id = pLocalMap->mvElementEdges[edgeIdx[i]].kf_id;
            int kf_idx = pLocalMap->mmKFID2KFindex.at(kf_id);

            Edge& edge = pLocalMap->mvKeyFrames[kf_idx]->mvEdges[kf_edge_idx];
            //-- 当前地图元素边缘的全局位姿
            Eigen::Matrix4d Trans_curr = pLocalMap->mvKeyFrames[kf_idx]->KF_pose_g.matrix();
            //-- 当前地图元素边缘相对于局部地图基准坐标系的相对位姿
            Eigen::Matrix4d Trans_ref_curr =  Trans_curr;
            //-- 单个边缘地图元素的点云
            std::vector<cv::Point3d> cloud;
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
                cv::Point3d point(points.x(),points.y(),points.z());
                cloud.push_back(point);
            }
            clusterCloud.insert(clusterCloud.end(), cloud.begin(), cloud.end());
        }
        clusterClouds.push_back(clusterCloud);
        clusterCloudColors.push_back(color);
    }
}

void visualizeMergedLocalMap(const edge_map::localMapPtr& pLocalMap,
                             std::vector<std::vector<cv::Point3d>>& mergedClouds)
{
    mergedClouds.clear();
    mergedClouds.reserve(pLocalMap->mvEleEdgeClusters.size());

    for(const auto& cluster : pLocalMap->mvEleEdgeClusters)
    {
        if(cluster.mbMerged == false) continue;
        std::vector<cv::Point3d> merged_cloud = cluster.mvMergedCloud_ref;
        mergedClouds.push_back(merged_cloud);
    }
}

bool shouldAddKeyFrame(const Eigen::Matrix4d& trans, float t_thres, float angle_thres)
{
    //-- 得到当前相对于关键帧的平移与旋转
    Eigen::Matrix3d R_bias = trans.block(0,0,3,3);
    Eigen::Vector3d t_bias = trans.block(0,3,3,1);
    Eigen::AngleAxisd rotation_vector(R_bias);  
    double theta = rotation_vector.angle() * 180 / M_PI; // 转换为角度 
    double translation = t_bias.norm();

    bool select = theta > angle_thres || translation > t_thres;
    return select;
}

void getSlidingWindow(const edge_map::localMapPtr& pLocalMap,
                      std::vector<Eigen::Matrix4d>& sliding_window)
{
    sliding_window.clear();
    for(size_t i = 0; i < pLocalMap->mvKeyFrames.size(); ++i)
    {
        sliding_window.push_back(pLocalMap->mvKeyFrames[i]->KF_pose_g.matrix());
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

    float angle_thres = ph.win.kf_rot_thres;
    float t_thres = ph.win.kf_trans_thres;

    double time_consume_total = 0.0;     //-- 实际用时
    double time_consume_technical = 0.0; //-- 30ms一帧的理论用时

    edgeSelector selector(20.0, ph.fine.canny_low, ph.fine.canny_high);

    std::vector<std::vector<cv::Point3d>> clusterClouds;
    std::vector<std::vector<cv::Point3d>> localMapClouds;
    std::vector<cv::Vec3b> clusterCloudColors;
    std::vector<Eigen::Matrix4d> slidingWindow;
    std::vector<std::vector<cv::Point3d>> environment_cloud;


    bool isFirstWindowOptimized = false;
    std::vector<Sophus::SE3d> list_poses_KF_origin;
    std::vector<Sophus::SE3d> list_poses_KF_optimized;

    std::vector<double> list_kf_stamps;
    std::vector<Eigen::Matrix4d> list_kf_poses;



    voViewer viewer("ROEVO");

    for(int i = 0; i < N; ++i)
    {
        //-- 得到当前帧的粗匹配位姿
        Eigen::Matrix4d pose_curr_coarse = tum_pose::getStaticGTPose(rgb_stamp_seq[i], coarse_stamps, coarse_poses);
        Eigen::Matrix3d R = pose_curr_coarse.block<3, 3>(0, 0);
        Eigen::Vector3d t = pose_curr_coarse.block<3, 1>(0, 3);
        Sophus::SE3d T(R, t);

        //-- 从这里开始记时，以30ms为周期发布数据
        auto start_timer = std::chrono::steady_clock::now();

        if(i == 0)
        {
            pose_last = pose_curr_coarse;
        }

        //-- 得到当前相对于关键帧的平移与旋转, 以判断是否需要加关键帧
        Eigen::Matrix4d trans = pose_last.inverse() * pose_curr_coarse;
        bool select = shouldAddKeyFrame(trans, t_thres, angle_thres);

        if (select || i == 0)
        {
            pose_last = pose_curr_coarse;

            //-- 获取当前的先验位姿
            Sophus::SE3d pose_curr;
            if(i == 0)
            {
                pose_curr = T;
            }else{
                Sophus::SE3d pose_last_prior = list_poses_KF_origin.back();
                Sophus::SE3d pose_bias = pose_last_prior.inverse() * T;
                Sophus::SE3d pose_last_adjust = pLocalMap->mvKeyFrames.back()->KF_pose_g;
                pose_curr = pose_last_adjust * pose_bias;
            }

            list_poses_KF_origin.push_back(T);

            cv::Mat imgRGB = cv::imread(path_tum + rgb_file_seq[i], CV_LOAD_IMAGE_COLOR);
            cv::Mat imgDepth = cv::imread(path_tum + depth_file_seq[i], CV_LOAD_IMAGE_UNCHANGED);
            imgDepth.convertTo(imgDepth, CV_32F, mDepthMapFactor);
            selector.processImage(imgRGB);

            //-- 根据帧信息与位姿信息创建一个关键帧
            KeyFramePtr pKF( new KeyFrame(i, pose_curr, rgb_stamp_seq[i], selector.mvEdges, imgRGB, imgDepth, ph.fx, ph.fy, ph.cx, ph.cy));
            pKF->getFineSampledPoints(ph.fine.sample_bias);
            
            std::cout<<"add frame "<< i << std::endl;
            pLocalMap->addFrame2LocalMap(pKF);

            visualizeAssociationResult(pLocalMap, clusterClouds, clusterCloudColors);

            if(pLocalMap->mvKeyFrames.size() == ph.win.window_size)
            {
                
                pLocalMap->clusterFittingProjection();
                edge_map::Optimizer::optimizeAllInvolvedKFs(pLocalMap);

                visualizeMergedLocalMap(pLocalMap, localMapClouds);

                // for(int j = 0; j < 7; ++j)
                // {
                //     double removed_stamp = pLocalMap->mvKeyFrames.front()->KF_stamp;
                //     Eigen::Matrix4d removed_pose = pLocalMap->mvKeyFrames.front()->KF_pose_g.matrix();
                    
                //     pLocalMap->removeKeyFrameFront();

                //     list_kf_poses.push_back(removed_pose);
                //     list_kf_stamps.push_back(removed_stamp);

                //     std::cout<<removed_stamp<<std::endl;
                // }

                for(int j = 0; j < ph.win.window_step; ++j)
                {
                    double removed_stamp = pLocalMap->mvKeyFrames[j]->KF_stamp;
                    Eigen::Matrix4d removed_pose = pLocalMap->mvKeyFrames[j]->KF_pose_g.matrix();

                    list_kf_poses.push_back(removed_pose);
                    list_kf_stamps.push_back(removed_stamp);
                }

                std::vector<KeyFramePtr> newKFs(pLocalMap->mvKeyFrames.begin()+ph.win.window_step, 
                                                pLocalMap->mvKeyFrames.begin()+ph.win.window_size);
                pLocalMap.reset(new edge_map::localMap());
                for(int j = 0; j < ph.win.window_size - ph.win.window_step; ++j)
                {
                    std::cout<<j<<std::endl;
                    newKFs[j]->mmEdgeIndex2ElementEdgeID.clear();
                    newKFs[j]->mmMapAssociations.clear();
                    pLocalMap->addFrame2LocalMap(newKFs[j]);
                }
            }
        }

        auto end_timer = std::chrono::steady_clock::now();
        auto dt = std::chrono::duration<double, std::milli>(end_timer - start_timer).count();
        
        time_consume_total += dt;
        time_consume_technical += 30;
        if(time_consume_total < time_consume_technical)
        {
            //-- 实际运行比理论运行快那就停一停，实际运行比理论运行慢则按实际运行时间
            double wait = time_consume_technical - time_consume_total;
            
            //-- 在这停顿
            //-- std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(wait));
            
            time_consume_total = 0.0;
            time_consume_technical = 0.0;
        }

        viewer.update_Trajectory(pose_curr_coarse);
        viewer.set_CameraPoses(pose_curr_coarse);
        viewer.set_gtPoses(pose_curr_coarse);
        viewer.update_covisibilityCloud(clusterClouds, clusterCloudColors);
        viewer.update_localMap(localMapClouds);

        std::vector<cv::Point3d> currentLocalMapCloud;
        for(size_t k = 0; k < localMapClouds.size(); ++k)
        {
            currentLocalMapCloud.insert(currentLocalMapCloud.end(), localMapClouds[k].begin(), localMapClouds[k].end());
        }
        environment_cloud.push_back(currentLocalMapCloud);
        if(environment_cloud.size()>=150)
        {
            environment_cloud.erase(environment_cloud.begin());
        }


        getSlidingWindow(pLocalMap, slidingWindow);
        viewer.update_sliding_window(slidingWindow);
        viewer.update_Environment(environment_cloud);
    }
    std::cout<<"finish adding"<<std::endl;
    viewer.stop();

   
    std::string storeFileName = path_tum + "posesKF.txt";
    tum_file::saveEvaluationFiles(storeFileName, list_kf_poses, list_kf_stamps);

    return 0;
}
