#ifndef VOVIEWER_H
#define VOVIEWER_H

#include "visualizerBase.h"
//pcl dependency
#include <pcl/console/parse.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transforms.h>

#include <pangolin/pangolin.h>

class voViewer: public VisualizerBase{

public:
    voViewer(std::string windowName);
    ~voViewer(){ if(!stop_) stop(); }

    void set_CameraPoses(Eigen::Matrix4d fine_pose)
    {
        cameraPose = fine_pose;
    }

    void set_gtPoses(Eigen::Matrix4d fine_pose)
    {
        gtPose = fine_pose;
    }

    void update_Trajectory(Eigen::Matrix4d fine_pose)
    {
        trajectory.push_back(fine_pose);
    }

    void update_covisibilityCloud(const std::vector<std::vector<cv::Point3d>>& clusterClouds,
                                  const std::vector<cv::Vec3b>& clusterCloudColors)
    {
        // 清空输出容器
        std::vector<cv::Point3d> covisibility_cloud_new;
        std::vector<cv::Vec3b>   covisibility_color_new;

        // 检查输入是否有效
        if (clusterClouds.size() != clusterCloudColors.size()) {
            std::cerr << "Warning: clusterClouds and clusterCloudColors size mismatch!" << std::endl;
            return;
        }

        // 预计算总点数，避免多次扩容
        size_t total_points = 0;
        for (const auto& cloud : clusterClouds) 
        {
            total_points += cloud.size();
        }
        covisibility_cloud_new.reserve(total_points);
        covisibility_color_new.reserve(total_points);

        // 合并点云和颜色
        for (size_t i = 0; i < clusterClouds.size(); ++i) 
        {
            const auto& cloud = clusterClouds[i];
            const auto& color = clusterCloudColors[i];
            
            covisibility_cloud_new.insert(covisibility_cloud_new.end(), cloud.begin(), cloud.end());
            covisibility_color_new.insert(covisibility_color_new.end(), cloud.size(), color);
        }

        covisibility_cloud = std::move(covisibility_cloud_new);
        covisibility_color = std::move(covisibility_color_new);

    }

    void update_TrajectoryGT(Eigen::Matrix4d fine_pose)
    {
        trajectory_GT.push_back(fine_pose);
    }

    void update_localMap(std::vector<std::vector<cv::Point3d>> local_map_cloud)
    {
        localMap_cloud = std::move(local_map_cloud);
    }

    void update_Environment(std::vector<std::vector<cv::Point3d>> pcloud)
    {
        environment_cloud = std::move(pcloud);
    }

    void update_sliding_window(std::vector<Eigen::Matrix4d> window)
    {
        sliding_window.clear();
        for(int i = 0; i < window.size(); ++i){
            sliding_window.push_back(window[i]);
        }
    }

    double get_trajectory_prop(){ return *slide_bar; }

protected:
    void start();
    void render_loop();

private:
    //交互视图所需材料
    Eigen::Matrix4d cameraPose; //-- 当前帧的相机位姿
    Eigen::Matrix4d gtPose; //-- 当前真值位姿，用于调整视角
    std::vector<Eigen::Matrix4d> trajectory; //-- 轨迹(fine tracking的完整轨迹)
    std::vector<Eigen::Matrix4d> trajectory_GT; //-- 轨迹(轨迹真值的完整轨迹)

    std::vector<Eigen::Matrix4d> sliding_window;
    
    std::vector<std::vector<cv::Point3d>> localMap_cloud;
    std::vector<std::vector<cv::Point3d>> environment_cloud;
    std::vector<cv::Point3d> covisibility_cloud;
    std::vector<cv::Vec3b>   covisibility_color;

    //控件部分
    std::shared_ptr<pangolin::Var<bool>>      follow;
    std::shared_ptr<pangolin::Var<bool>>      show_covisibility;
    std::shared_ptr<pangolin::Var<double>> slide_bar;


    std::string windowName_;

};

#endif