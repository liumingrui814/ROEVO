#ifndef EDGECLUSTER_H
#define EDGECLUSTER_H

#include <iostream>
#include <fstream>
#include <queue>
#include <sys/types.h>
#include <dirent.h>
#include <map>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <Eigen/Dense>
#include <Eigen/Core>

#include <chrono>

/**
 * @brief 无序的边缘点，仅在边缘组织的过程中使用到，
 * @details 存储度、当前ID与父节点ID等信息，通过拓展的拓扑关系将canny散点组织成有序边缘
*/
struct edgePoint{
    //-- 边缘点的像素坐标
    cv::Point pixel;
    //-- 边缘点的父节点的ID
    int father_id;
    //-- 边缘点自己的ID
    int point_id;
    //-- 边缘点的入度（该节点是几个点的父节点）
    int degree;
    //-- 是否是区域生长的遍历起点（root）
    bool isRoot;

    edgePoint(cv::Point _pixel, int _point_id, int _father_id)
    {
        pixel = _pixel;
        father_id = _father_id;
        point_id = _point_id;
        degree = 0;
        isRoot = false;
    }
};

/**
 * @brief 无序的边缘聚类，存储一组无序但可以自组织的edgePoint，可实现edgePoint的自组织
 * @details 存储edgePoint, 利用edgePoint中的拓扑关系进行边缘的自组织
*/
class EdgeCluster{
public:
    //-- 每个边缘的点列表，可以用push_back进行操作
    std::vector<edgePoint> mvPoints;
    //-- 每个边缘点的标识ID与边缘点在points中的索引的映射关系
    std::map<int, int> mmIndexMap;

    EdgeCluster(){}

    EdgeCluster(std::vector<edgePoint> list);

    //-- 整理边缘，根据区域生长建立的邻接关系将边缘整理成有序的
    std::vector<edgePoint> organize();

private:
    //-- 计算每个点的入度（即该节点是几个点的父节点）
    void calculateDegree();


};

#endif