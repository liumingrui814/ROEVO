#include <iostream>
#include <fstream>
#include <sophus/se3.hpp>
#include <stdio.h>

#include <chrono> //-- 计时函数

#include "edgeSelector.h"
#include "Frame.h"

#include "paramLoader.h"

#include "tum_pose.h"
#include "tum_file.h"

#include "visualizerPCL.h"

cv::Mat visualizeEdgesRaw(cv::Mat imgBackGround, std::vector<Edge>& edges)
{
    cv::Mat image_viz = imgBackGround.clone();

    if(image_viz.channels() == 3){
        cv::cvtColor(image_viz, image_viz, cv::COLOR_BGR2GRAY);
        for (int y = 0; y < image_viz.rows; ++y) {
            for (int x = 0; x < image_viz.cols; ++x) {
                // 获取当前像素值
                uchar& pixel = image_viz.at<uchar>(y, x);
                
                // 将像素值减半，确保不会小于0
                pixel = static_cast<uchar>(pixel / 1.3);
            }
        }
    }
    cv::cvtColor(image_viz, image_viz, cv::COLOR_GRAY2BGR);
    int maxLabel = 0;
    for(int i = 0; i < edges.size(); ++i){
        for(int j = 0; j < edges[i].mvPoints.size(); ++j){
            orderedEdgePoint curr = edges[i].mvPoints[j];
            cv::circle(image_viz, cv::Point(curr.x, curr.y), 1, cv::Scalar(0,200,0), 1, cv::LINE_AA);
        }
    }
    return image_viz;
}

cv::Mat visualizeEdgesOrient(cv::Mat imgBackGround, std::vector<Edge>& edges)
{
    cv::Mat hsvTabel(1,180,CV_8UC3,cv::Scalar::all(0));
    for(int i = 0; i < 180; i++){
        hsvTabel.at<cv::Vec3b>(0,i) = cv::Vec3b(i,255,255);
    }
    cv::cvtColor(hsvTabel, hsvTabel, cv::COLOR_HSV2BGR);

    cv::Mat image_viz = imgBackGround.clone();

    if(image_viz.channels() == 3){
        cv::cvtColor(image_viz, image_viz, cv::COLOR_BGR2GRAY);
        for (int y = 0; y < image_viz.rows; ++y) {
            for (int x = 0; x < image_viz.cols; ++x) {
                // 获取当前像素值
                uchar& pixel = image_viz.at<uchar>(y, x);
                
                // 将像素值减半，确保不会小于0
                pixel = static_cast<uchar>(pixel / 1.3);
            }
        }
    }
    cv::cvtColor(image_viz, image_viz, cv::COLOR_GRAY2BGR);
    int maxLabel = 0;
    for(int i = 0; i < edges.size(); ++i){
        for(int j = 0; j < edges[i].mvPoints.size(); ++j){
            orderedEdgePoint curr = edges[i].mvPoints[j];
            float angle = curr.imgGradAngle;
            int ratio = int(angle / 2.0);
            cv::Vec3b color = hsvTabel.at<cv::Vec3b>(0,ratio);
            cv::circle(image_viz, cv::Point(curr.x, curr.y), 1, color, 1, cv::LINE_AA);
        }
    }
    return image_viz;
}

cv::Mat visualizeEdges(cv::Mat imgBackGround, std::vector<Edge>& edges)
{
    cv::RNG rng(66);
    cv::Mat image_viz = imgBackGround.clone();

    if(image_viz.channels() == 3){
        cv::cvtColor(image_viz, image_viz, cv::COLOR_BGR2GRAY);
        for (int y = 0; y < image_viz.rows; ++y) {
            for (int x = 0; x < image_viz.cols; ++x) {
                // 获取当前像素值
                uchar& pixel = image_viz.at<uchar>(y, x);
                
                // 将像素值减半，确保不会小于0
                pixel = static_cast<uchar>(pixel / 1.3);
            }
        }
    }
    cv::cvtColor(image_viz, image_viz, cv::COLOR_GRAY2BGR);
    int maxLabel = 0;
    for(int i = 0; i < edges.size(); ++i){
        int b = rng.uniform(0, 255);
        int g = rng.uniform(0, 255);
        int r = rng.uniform(0, 255);
        cv::Vec3b color = cv::Vec3b(b,g,r);
        //if(mvEdgeClusters[i].mvPoints.size()<15) continue;
        for(int j = 0; j < edges[i].mvPoints.size(); ++j){
            orderedEdgePoint curr = edges[i].mvPoints[j];
            image_viz.at<cv::Vec3b>(curr.y,curr.x) = color;
            cv::circle(image_viz, cv::Point(curr.x, curr.y), 1, color, 1, cv::LINE_AA);
        }
    }
    return image_viz;
}

cv::Mat visualizeEdgesOrganized(cv::Mat imgBackGround, std::vector<Edge>& edges)
{
     cv::Mat valueTabel(256,1,CV_8UC1);
    cv::Mat ColorTabel;
    for(int i = 0; i<256; i++){
        valueTabel.at<uint8_t>(i,0)=i;
    }
    cv::applyColorMap(valueTabel,ColorTabel,cv::COLORMAP_PARULA);
    cv::Mat image_viz = imgBackGround.clone();
    if(image_viz.channels() == 3){
        cv::cvtColor(image_viz, image_viz, cv::COLOR_BGR2GRAY);
        for (int y = 0; y < image_viz.rows; ++y) {
            for (int x = 0; x < image_viz.cols; ++x) {
                // 获取当前像素值
                uchar& pixel = image_viz.at<uchar>(y, x);
                
                // 将像素值减半，确保不会小于0
                pixel = static_cast<uchar>(pixel / 1.3);
            }
        }
    }
    cv::cvtColor(image_viz, image_viz, cv::COLOR_GRAY2BGR);
    int maxLabel = 0;
    for(int i = 0; i < edges.size(); ++i){
        //if(mvEdgeClusters[i].mvPoints.size()<15) continue;
        for(int j = 0; j < edges[i].mvPoints.size(); ++j){
            float proportion = float(j)/float(edges[i].mvPoints.size());
            int idx = cvRound(proportion * 255);
            orderedEdgePoint curr = edges[i].mvPoints[j];
            cv::Vec3b color = ColorTabel.at<cv::Vec3b>(idx,0);
            image_viz.at<cv::Vec3b>(curr.y,curr.x) = color;
            cv::circle(image_viz, cv::Point(curr.x, curr.y), 1, color, 1, cv::LINE_AA);
        }
    }
    return image_viz;
}

int main(int argc, char **argv) {
    std::string image_dir = "../data/";
    std::string image_name = argv[1];
    std::string path_image = image_dir + image_name;

    edgeSelector selector(20.0, 50, 100);

    cv::namedWindow("O-EDGE", cv::WINDOW_NORMAL);

    cv::Mat imgRGB_1 = cv::imread(path_image, CV_LOAD_IMAGE_COLOR);
    selector.processImage(imgRGB_1);

    cv::Mat imgSeg = visualizeEdges(imgRGB_1, selector.mvEdges);
    cv::Mat imgSeg_2 = visualizeEdgesOrganized(imgRGB_1, selector.mvEdges);
    cv::Mat imgSeg_3 = visualizeEdgesRaw(imgRGB_1, selector.mvEdges);
    cv::Mat imgSeg_4 = visualizeEdgesOrient(imgRGB_1, selector.mvEdges);

    // 创建一个大图像来存放四宫格
    cv::Mat collage(imgRGB_1.rows * 2, imgRGB_1.cols * 2, imgRGB_1.type());
    
    // 将四个图像放置到四宫格的相应位置
    // 左上角
    imgSeg_3.copyTo(collage(cv::Rect(0, 0, imgRGB_1.cols, imgRGB_1.rows)));
    // 右上角
    imgSeg_4.copyTo(collage(cv::Rect(imgRGB_1.cols, 0, imgRGB_1.cols, imgRGB_1.rows)));
    // 左下角
    imgSeg.copyTo(collage(cv::Rect(0, imgRGB_1.rows, imgRGB_1.cols, imgRGB_1.rows)));
    // 右下角
    imgSeg_2.copyTo(collage(cv::Rect(imgRGB_1.cols, imgRGB_1.rows, imgRGB_1.cols, imgRGB_1.rows)));
    
    // 显示四宫格
    cv::imshow("O-EDGE", collage);
    cv::waitKey(0); // 等待按键



    cv::destroyAllWindows();

    return 0;
}
