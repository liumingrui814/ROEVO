#ifndef LOCAL_MAP_H
#define LOCAL_MAP_H

#include "KeyFrame.h"
#include "disjointSet.h"
#include "edgeSelector.h"
#include "elementEdge.h"
#include <pcl/common/distances.h>

//-- 局部地图与帧的关联结果
typedef std::pair<pcl::PointCloud<pcl::PointXYZ>, std::vector<Edge>> match3d_2d;

namespace edge_map{

class localMap{
public:
    //-- 地图由一组边缘构成
    std::vector<elementEdge> mvElementEdges;
    //-- 地图的 elementEdge 的 element_id 与其在 mvElementEdges 中的索引的映射
    std::map<unsigned int, int> mmElementID2index;

    //-- 边缘地图元素与类别ID的映射, 
    //-- first是map element的label，second是map element的列表，即一团map element是一个完整边缘
    std::map<unsigned int, std::vector<int>> mmElementEdgeMap;

    //-- 对应地图元素ID对应的拟合边缘
    std::map<int, pcl::PointCloud<pcl::PointXYZ>> mmMergeEdgeMap;

    //-- 组成这个local(global) map的帧以及各个帧的位姿
    std::vector<KeyFramePtr> mvKeyFrames;
    std::map<int, int> mmKFID2KFindex;

    //-- 当前局部地图的基准坐标系位姿
    Sophus::SE3d T_ref;

    //-- 使用参考关键帧初始化一个map类
    void initWithRefFrame(KeyFramePtr frame_ref);

    //-- 在有两个关键帧在局部地图中时初始化一个局部地图，该局部地图只存储两帧共视的边缘特征
    void initLocalMap();

    void addNewFrame(KeyFramePtr frame_cur);

    //-- 得到label与地图边缘元素的映射
    void generateLabelMap();

private:
    //-- 以下函数是一些并查集的操作

    //-- 找到并查集的根节点
    unsigned int findRoot(unsigned int curr_id);

    //-- 并查集剪枝，让所有节点都直接指向根节点
    void pruningMap();

    //-- 基于多对多的关联策略关联两个关键帧
    associationResult associationMulti2Multi(KeyFramePtr frame_ref, KeyFramePtr frame_cur);

};

}

#endif