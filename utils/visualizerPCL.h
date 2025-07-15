/**
 * @brief 基于PCL的可视化库，
 * @author 猎魔人
 * @date 2024-9-22
 * @details 参考ORB-SLAM2中回环检测的代码部分
*/
#ifndef VISUALIZERPCL_H
#define VISUALIZERPCL_H
#include <iostream>
#include <fstream>
#include <sstream> 
#include <string>
#include <sophus/se3.hpp>

#include <pcl/console/parse.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transforms.h>
// #include <pcl/filters/voxel_grid.h>
// #include <pcl/kdtree/kdtree_flann.h>
// #include <pcl/octree/octree_search.h>

#include <Eigen/Core>
#include <Eigen/Dense>

namespace viz_util{

template <typename pointType>
void drawPointCloudColor(boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer,
                         std::string cloud_identity, boost::shared_ptr<pcl::PointCloud<pointType>> cloud,
                         unsigned int r, unsigned int g, unsigned int b, unsigned int point_size)
{
    pcl::visualization::PointCloudColorHandlerCustom<pointType> colorHandler(cloud, r, g, b);
    viewer->addPointCloud<pointType> (cloud, colorHandler, cloud_identity);
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, point_size, cloud_identity);
}

template <typename pointType>
void drawPointCloudRandomColor(boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer,
                         std::string cloud_identity, boost::shared_ptr<pcl::PointCloud<pointType>> cloud,
                         unsigned int point_size)
{
    pcl::visualization::PointCloudColorHandlerRandom<pointType> colorHandler(cloud);
    viewer->addPointCloud<pointType> (cloud, colorHandler, cloud_identity);
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, point_size, cloud_identity);
}

template <typename pointType>
void drawPointCloudChannel(boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer,
                           std::string cloud_identity, boost::shared_ptr<pcl::PointCloud<pointType>> cloud,
                           std::string channel, unsigned int point_size)
{
    if(!(channel == std::string("x") || channel == std::string("y") || 
         channel == std::string("z") || channel == std::string("intensity")))
    {
        std::cout<<"\033[31m [VISUALIZATION ERROR] \033[0m" << ": invalid pointcloud channel name"<<std::endl;
    }
    pcl::visualization::PointCloudColorHandlerGenericField<pointType> colorHandler(cloud, channel);
    viewer->addPointCloud<pointType> (cloud, colorHandler, cloud_identity);
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, point_size, cloud_identity);
}

template <typename pointType>
void drawLine(boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer,
              std::string line_identity, pointType point_1, pointType point_2,
              unsigned int r, unsigned int g, unsigned int b, unsigned int line_size)
{
    float r_norm = float(r) / 255.0;
    float g_norm = float(g) / 255.0;
    float b_norm = float(b) / 255.0;
    viewer->addLine<pointType>(point_1, point_2, r_norm, g_norm, b_norm, line_identity);
    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, line_size, line_identity);
}

void drawLineEigen(boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer,
              std::string line_identity, Eigen::Vector3d point_1, Eigen::Vector3d point_2,
              unsigned int r, unsigned int g, unsigned int b, unsigned int line_size)
{
    float r_norm = float(r) / 255.0;
    float g_norm = float(g) / 255.0;
    float b_norm = float(b) / 255.0;
    pcl::PointXYZ pt_1_pcl(point_1.x(), point_1.y(), point_1.z());
    pcl::PointXYZ pt_2_pcl(point_2.x(), point_2.y(), point_2.z());
    viewer->addLine<pcl::PointXYZ>(pt_1_pcl, pt_2_pcl, r_norm, g_norm, b_norm, line_identity);
    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, line_size, line_identity);
}

void drawCamera(boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer,
                Eigen::Matrix4d cameraPose, std::string camera_id, double camSize,
                unsigned int r, unsigned int g, unsigned int b,
                unsigned int line_size)
{
    //画相机
    const float w = camSize / 2.0;
    const float h = w * 0.75;
    const float z = w * 0.6;
    //使用位置变换矩阵
    Eigen::Matrix3d R = cameraPose.block(0,0,3,3);
    Eigen::Vector3d t = cameraPose.block(0,3,3,1);
    
    //-- 画线
    drawLineEigen(viewer, camera_id + "cam_line_1", 
             R * Eigen::Vector3d(0,0,0) + t,   R * Eigen::Vector3d(w,h,z) + t,
             r, g, b, line_size);
    drawLineEigen(viewer, camera_id + "cam_line_2", 
             R * Eigen::Vector3d(0,0,0) + t,   R * Eigen::Vector3d(-w,-h,z) + t, 
             r, g, b, line_size);
    drawLineEigen(viewer, camera_id + "cam_line_3", 
             R * Eigen::Vector3d(0,0,0) + t,   R * Eigen::Vector3d(-w,h,z) + t, 
             r, g, b, line_size);
    drawLineEigen(viewer, camera_id + "cam_line_4", 
             R * Eigen::Vector3d(0,0,0) + t,   R * Eigen::Vector3d(w,-h,z) + t, 
             r, g, b, line_size);
    drawLineEigen(viewer, camera_id + "cam_line_5", 
             R * Eigen::Vector3d(w,h,z) + t,   R * Eigen::Vector3d(w,-h,z) + t, 
             r, g, b, line_size);
    drawLineEigen(viewer, camera_id + "cam_line_6", 
             R * Eigen::Vector3d(-w,h,z) + t,  R * Eigen::Vector3d(-w,-h,z) + t, 
             r, g, b, line_size);
    drawLineEigen(viewer, camera_id + "cam_line_7", 
             R * Eigen::Vector3d(-w,h,z) + t,  R * Eigen::Vector3d(w,h,z) + t, 
             r, g, b, line_size);
    drawLineEigen(viewer, camera_id + "cam_line_8", 
             R * Eigen::Vector3d(-w,-h,z) + t, R * Eigen::Vector3d(w,-h,z) + t, 
             r, g, b, line_size);
}






}
#endif