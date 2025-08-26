// * Key-Frame
//-- KeyFrame contains 3D information + pose information + time information + co-visibility information

#ifndef KEYFRAME_H
#define KEYFRAME_H

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

//dependency for sophus
#include <sophus/se3.hpp>

#include <Eigen/Dense>
#include <Eigen/Core>

#include <chrono>
#include <numeric> 
#include <random>

#include "edge.h"
#include "edgeSelector.h"
#include "disjointSet.h"

//-- 关联的结果：first是cur帧的边缘与ref帧关联边缘的idx映射
//--           second是ref帧的边缘对应的类别（ref帧的不同边缘可能是同一类）
typedef std::pair<std::unordered_map<int, int>, std::unordered_map<int, int>> associationResult;


class KeyFrame
{
public:
    //-- index of the frame
    int KF_ID;

    //-- 关键帧的时间戳
    double KF_stamp;
    //-- 关键帧的全局位姿
    Sophus::SE3d KF_pose_g;

    //-- 构成一帧图像的所有边缘特征
    std::vector<Edge> mvEdges;
    //-- 边缘ID与mvEdge索引的映射关系
    std::map<int, int> mmIndexMap;

    //-- 灰度图像，用于构造 fine-tracking 的光度误差等
    cv::Mat mMatGray;

    //-- 图像大小，整个序列都是唯一的
    int mWidth;
    int mHeight;

    //-- 图像内参
    float mFx, mFy, mCx, mCy;

    //-- 用于半径搜索的二维Mat
    cv::Mat mMatSearch;

    //-- 关键帧的第i条（索引）边对应的局部地图中的elementEdge的ID
    std::map<int, unsigned int> mmEdgeIndex2ElementEdgeID;
    //-- 预存储的关键帧与其他关键帧的多对多关联关系, first是对应关键帧的ID，second是关联结果
    std::map<int, associationResult> mmMapAssociations;

    KeyFrame(){}

    KeyFrame(int ID, Sophus::SE3d pose, double stamp, std::vector<Edge> vEdges, const cv::Mat& matRGB, const cv::Mat& matDepth,
          const float& fx, const float& fy, const float& cx, const float& cy);

    void searchRadius(float x, float y, double radius, std::vector<orderedEdgePoint>& result);
    
    bool isPointsAssociated(const orderedEdgePoint& pt1, const orderedEdgePoint& pt2);

    bool isPointsAssociatedAlign(const orderedEdgePoint& warped_pt, 
                                 const orderedEdgePoint& neighbor_pt, 
                                 const float& depth_warp);
    
    //-- 考虑需要将query_edge重投影的情况下获取边缘级别的关联，返回对应关联的边缘的索引
    std::vector<int> edgeWiseCorrespondenceReproject(Edge& query_edge, const Sophus::SE3d& T2curr);

    //-- 在 local mapping 的阶段进行边缘关联，需要考虑当前关键帧与局部地图的交互
    std::vector<int> edgeWiseCorrespondenceLocalMapping(Edge& query_edge, const Sophus::SE3d& T2curr);
    
    //-- 可视化搜索平面
    cv::Mat visualizeSearchPlain();
    
    //-- 获取粗匹配所需的3D边缘特征点
    std::vector<orderedEdgePoint> getCoarseSampledPoints(int bias, int maximum_point);

    void getFineSampledPoints(int bias);

private:

    //-- 在整理过边缘后遍历边缘中的边缘点，更新索引关系
    void assignPropertyIdx();
    
    //-- 根据边缘信息创建二维的搜索阵列，用于半径邻域搜索
    void constructSearchPlain();

    //-- 并行创建二维搜索阵列
    void constructSearchPlainParallel();

    //-- 遍历边缘中的每一个点计算：远近可信分数，观测可信分数 以及 深度补全
    void assignProperty3D(const cv::Mat& matDepth);

    //-- 为边缘的单个点计算 远近可信分数，观测可信分数 以及 深度补全
    void assignProperty3DEach(orderedEdgePoint& pt, const cv::Mat& matDepth);


    //-- 对边缘进行调整，祛除深度无效的边缘，以及祛除整体无效的边缘
    void edgeCullingDepth();
    void edgeCullingDepthParallel();

    //-- 对边缘进行调整，切割与重组深度不一致的边缘，以及祛除整体不一致性较强的边缘
    void edgeCullingContinuity();
    
    //-- 对边缘进行调整，较远或包含较模糊观测的边缘需要被祛除
    void edgeCullingQuality();
    void edgeCullingQualityParallel();
};


//-- 用using 定义智能指针
using KeyFramePtr = std::shared_ptr<KeyFrame>;

#endif