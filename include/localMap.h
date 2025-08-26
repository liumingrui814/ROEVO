#ifndef LOCAL_MAP_H
#define LOCAL_MAP_H

#include "KeyFrame.h"
#include "disjointSet.h"
#include "elementEdge.h"
#include "featureMerger.h"
#include <pcl/common/distances.h>

#include <pcl/console/parse.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/octree/octree_search.h>

//-- 局部地图与帧的关联结果
typedef std::pair<std::vector<cv::Point3d>, std::vector<edge_map::elementEdge>> match3d_2d;

namespace edge_map{

class localMap{
public:

    enum class State {
        NOT_INITIALIZED,  //-- 未初始化 localmap
        INITIALIZED,      //-- 已初始化 localmap (两帧)
        LOST              //-- 局部地图和最新关键帧关联不上
    };

    State msState;

    //-- 地图由一组边缘构成
    std::vector<elementEdge> mvElementEdges;
    //-- 地图的 elementEdge 的 element_id 与其在 mvElementEdges 中的索引的映射
    std::map<unsigned int, int> mmElementID2index;

    
    //-- edge cluster 的数据类型，由mvEleEdgeClusters存储，由 mmClusterID2index 索引
    std::vector<elementEdgeCluster> mvEleEdgeClusters;
    std::map<unsigned int, int> mmClusterID2index;

    //-- 组成这个local(global) map的帧以及各个帧的位姿
    std::vector<KeyFramePtr> mvKeyFrames;
    std::map<int, int> mmKFID2KFindex;

    //-- 当前局部地图的基准坐标系位姿
    //Sophus::SE3d T_ref;

    //-- 在有两个关键帧在局部地图中时初始化一个局部地图，该局部地图只存储两帧共视的边缘特征
    void initLocalMap();

    void addFrame2LocalMap(KeyFramePtr frame_cur);

    //-- 删除 mvKeyFrames 队列中的**第一个**关键帧
    void removeKeyFrameFront();

    //-- 在 3D 空间中拟合所有的cluster，得到用于 local mapping / BA 的整合边
    void clustersFitting3D();

    //-- 依靠重投影和极线判定拟合所有的 cluster，得到用于 local mapping / BA 的整合边
    void clusterFittingProjection();

    //-- 获得当前的局部地图与某个关键帧的关联关系
    void getAssoFrameMergeEdge(int kf_id_dst, std::vector<match3d_2d>& matches, std::vector<double>& weights);

    localMap()
    {
        //-- cluster的配色器的初始化
        elementEdgeCluster::initRandom();
        //-- 未初始化
        msState = State::NOT_INITIALIZED;
    }

private:
    //-- 以下函数是一些并查集的操作

    //-- 找到并查集的根节点
    unsigned int findRoot(unsigned int curr_id);

    //-- 并查集剪枝，让所有节点都直接指向根节点
    void pruningMap();

    void mergeElementCluster(int cluster_idx_1, int cluster_idx_2);

    //-- 根据每个 cluster 的 count_not_update 来决定是否删掉cluster
    void elementClusterCulling();

    //-- 基于多对多的关联策略关联两个关键帧
    associationResult associationMulti2Multi(KeyFramePtr frame_ref, KeyFramePtr frame_cur);

    //-- 合并单个 cluster 的点云，得到融合后的点云
    void getMergedCluster(const std::vector<pcl::PointCloud<pcl::PointXYZ>>& clusterCloud,
                                      pcl::PointCloud<pcl::PointXYZ>& mergedCloud);

    
    pcl::PointXYZ calcCloudCentroid(const pcl::PointCloud<pcl::PointXYZ>& pointCloud, pcl::PointXYZ current);

    void assignWeights();

};

//-- 用using 定义智能指针
using localMapPtr = std::shared_ptr<localMap>;

}

#endif