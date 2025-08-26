#ifndef ELEMENT_EDGE_H
#define ELEMENT_EDGE_H

#include "KeyFrame.h"
#include "disjointSet.h"

#include <random>

namespace edge_map{

//-- 用于多帧关联的边缘Edge的套壳
class elementEdge{
    public:
        static unsigned int id_counter; 
        unsigned int element_id;   //-- 边缘自己的ID
        int union_id;              //-- 并查集元素，表示该边缘与谁同类

        //-- 边缘的关键帧索引与关键帧数据索引
        int kf_id;
        int kf_edge_idx;

        elementEdge(){}
        
        elementEdge(int _frame_id, int _frame_edge_idx):
            element_id(id_counter++),          //-- 用当前element_id值初始化id，然后递增
            union_id(-1),                      //-- 并查集 -1 表示根节点
            kf_id(_frame_id),
            kf_edge_idx(_frame_edge_idx)
        {}
};

class elementEdgeCluster{
public:
    //-- 在并查集剪枝后，根节点的element_id 即为 cluster_id, 确保不重复
    unsigned int cluster_id;

    //-- 在新加入了 n 帧后均未更新这个Cluster
    int count_not_update;
    bool mbModifiedCur;

    //-- 仅在可视化的时候使用，每个cluster有唯一的可视化颜色
    cv::Vec3b visColor;
    //-- 用于随机赋值颜色
    static std::mt19937 rng; // 随机数生成器
    static std::uniform_int_distribution<unsigned char> dist; // 分布器

    std::vector<unsigned int> mvElementEdgeIDs;

    bool mbMerged;
    double dist_thres;
    double weightBA;
    std::vector<cv::Point3d> mvMergedCloud_ref;
    std::vector<int> mvInvolvedLocalMapElementIndices;

    
    elementEdgeCluster(unsigned int id, std::vector<unsigned int> eleEdges)
    {
        cluster_id = id;
        mvElementEdgeIDs = std::move(eleEdges);
        count_not_update = 0;
        //-- 随机一个颜色
        visColor = cv::Vec3b(dist(rng), dist(rng), dist(rng));
        //-- 被构造也算最新操作，因此用true初始化
        mbModifiedCur = true;

        mbMerged = false;

        weightBA = 0.0;
    }

    // 静态初始化（类外调用 elementEdgeCluster：：initRandom()）
    static void initRandom() {
        rng.seed(std::random_device{}());
    }
};

}

#endif