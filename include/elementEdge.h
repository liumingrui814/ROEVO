#ifndef ELEMENT_EDGE_H
#define ELEMENT_EDGE_H

#include "KeyFrame.h"
#include "disjointSet.h"

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

    elementEdge(int _frame_id, int _frame_edge_idx)
    : element_id(id_counter++),          //-- 用当前element_id值初始化id，然后递增
      union_id(-1),                      //-- 并查集 -1 表示根节点
      kf_id(_frame_id),
      kf_edge_idx(_frame_edge_idx)
    {}
};

}

#endif