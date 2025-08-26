#ifndef EDGE_H
#define EDGE_H

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
 * @brief 有序的边缘点，是直接参与到slam过程中的边缘点类型
 * @details 不再存储与相邻有关的信息，仅存储当前位置的图像属性与结构属性
*/
class orderedEdgePoint{
public:
    //########################### 
    //--      2D相关成员
    //########################### 

    //-- 像素坐标
    double x;
    double y;
    //-- RGBD给的深度
    float depth;
    //-- 梯度角度
    float imgGradAngle;

    //########################### 
    //--      3D相关成员
    //########################### 
    double x_3d;
    double y_3d;
    double z_3d;
    float score_depth;
    float score_visible;
    

    //########################### 
    //--      索引相关成员
    //###########################
    int frame_edge_ID;
    int frame_point_index;

    //########################### 
    //--      关联相关成员
    //###########################
    bool mbAssociated;
    //-- 半径内搜索得到的最近邻列表
    std::vector<int> mvAssoFrameEdgeIDs;
    std::vector<int> mvAssoFramePointIndices;
    int asso_edge_ID; //-- 与当前点关联的参考帧点的edge ID
    int asso_point_index; //-- 与当前点关联的参考帧点的point index

    orderedEdgePoint(double _x, double _y, float _imgGradAngle){
        x = _x;
        y = _y;
        imgGradAngle = _imgGradAngle;
        mbAssociated = false;
    }
};

/**
 * @brief 自组织之后的有序边缘，存储有序的orderedEdgePoint, 其索引就代表其顺序
 * @details 本类直接用于slam过程中，直接参与特征的匹配、关联以及配准优化
*/
class Edge{
public:
    int edge_ID;

    //-- 每个边缘的点列表，可以用push_back进行操作
    std::vector<orderedEdgePoint> mvPoints;
    float mVisScore;

    //-- 采样之后的索引，在mvPoints中检索，与edgeCloud一一对应
    std::vector<int> mvSampledEdgeIndex;

    void samplingEdgeUniform(int bias);

    Edge(){}

    Edge(int ID){ edge_ID = ID;}

    Edge(int ID, std::vector<orderedEdgePoint> list);

    void push_back(orderedEdgePoint pt);

    //-- 检查两个边缘是否可以建立关联，如果可以建立关联则返回true, 不行则返回false
    bool isAssociatedWith(Edge query_edge);

    void calcEdgeScoreViz();

};

#endif