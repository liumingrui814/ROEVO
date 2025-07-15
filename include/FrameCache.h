#ifndef FRAME_H
#define FRAME_H

#include <iostream>
#include <fstream>
#include <queue>
#include <sys/types.h>
#include <dirent.h>
#include <map>
#include <unordered_set>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/opencv.hpp>

#include <pcl/console/parse.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/octree/octree_search.h>

#include <Eigen/Dense>
#include <Eigen/Core>

#include <chrono>

//dependency for sophus
#include <sophus/se3.hpp>

#include "edge.h"
#include "edgeSelector.h"
#include "disjointSet.h"

class Frame{
public:
    int frame_ID;

    // edgeSelector selector; //-- 边缘检测器决定不予存储，留着没用
    cv::Mat mMatDepth;

    //-- 用于半径搜索的二维Mat
    cv::Mat mMatSearch;

    //-- 构成一帧图像的所有边缘特征
    std::vector<Edge> mvEdges;
    //-- 边缘ID与mvEdge索引的映射关系
    std::map<int, int> mmIndexMap;

    //-- 粗优化与精优化分别利用的边缘点
    std::vector<orderedEdgePoint> mvCoarseFeaturePoints;
    std::vector<orderedEdgePoint> mvFineFeaturePoints;
    // bool isCloudGenerated;

    // //-- 精匹配点在图像坐标系下的点云以及搜索树(2D像素坐标的搜索树)，
    // //-- mFrameCloudFine2D与mvFineFeaturePoints在索引上完全对应
    // pcl::PointCloud<pcl::PointXYZ>::Ptr mpFrameCloudFine2D;
    // pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;

    //-- 相机内参，帧内作为备份
    double mFx;
    double mFy;
    double mCx;
    double mCy;
    //-- 图像大小，整个序列都是唯一的
    int mWidth;
    int mHeight;

    Frame(){}

    Frame(const Frame& frame) = default;

    Frame(Frame& frame){
        frame_ID = frame.frame_ID;
        mMatDepth = frame.mMatDepth.clone();
        mvEdges = frame.mvEdges;
        mmIndexMap = frame.mmIndexMap;
        // if(frame.isCloudGenerated == true){
        //     isCloudGenerated = true;
        //     mpFrameCloudFine2D.reset(new pcl::PointCloud<pcl::PointXYZ>());
        //     *mpFrameCloudFine2D = *(frame.mpFrameCloudFine2D);
        // }
        // kdtree = frame.kdtree;
        mFx = frame.mFx;
        mFy = frame.mFy;
        mCy = frame.mCy;
        mCx = frame.mCx;
        mWidth = frame.mWidth;
        mHeight = frame.mHeight;
        mvCoarseFeaturePoints = frame.mvCoarseFeaturePoints;
        mvFineFeaturePoints = frame.mvFineFeaturePoints;
    }

    Frame(int ID, cv::Mat imgDepth, double fx, double fy, double cx, double cy, std::vector<Edge> vEdges);

    Frame clone(){
        Frame copy_frame(*this);
        return copy_frame;
    }

    // KNNresult searchRadius(float x, float y, double radius);
    void searchRadius(float x, float y, double radius, std::vector<orderedEdgePoint>& result);

    //-- 考虑需要将query_edge重投影的情况下获取边缘级别的关联，返回对应关联的边缘的索引
    std::vector<int> edgeWiseCorrespondenceReproject(Edge& query_edge, Eigen::Matrix4d T2curr);

    //-- 获得距离限制采样后的用于粗优化直接法的边缘点列
    std::vector<orderedEdgePoint> getCoarseSampledPoints(int bias, int maximum_point); 

    //-- 获得基于曲率采样后的用于精优化几何-光度优化的边缘点列
    std::vector<orderedEdgePoint> getFineSampledPoints(int bias);

private:

    //-- 遍历边缘中的每一个点计算：远近可信分数，观测可信分数 以及 深度补全
    void assignProperty3D();

    //-- 在整理过边缘后遍历边缘中的边缘点，更新索引关系
    void assignPropertyIdx();

    //-- 为边缘的单个点计算 远近可信分数，观测可信分数 以及 深度补全
    void assignProperty3DEach(orderedEdgePoint& pt);

    //-- 基于旧方法（取小的块的深度）计算点的深度
    float assignPriorDepthPatch(orderedEdgePoint& pt);

    //-- 对边缘进行调整，祛除深度无效的边缘，以及祛除整体无效的边缘
    void edgeCullingDepth();

    //-- 对边缘进行调整，切割与重组深度不一致的边缘，以及祛除整体不一致性较强的边缘
    void edgeCullingContinuity();

    //-- 根据边缘信息创建二维的搜索阵列，用于半径邻域搜索
    void constructSearchPlain();
};


//-- 用using 定义智能指针
using FramePtr = std::shared_ptr<Frame>;

#endif