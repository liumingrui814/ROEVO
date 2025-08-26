/** 
 * @brief 基于光度误差的雅克比矩阵计算
 * @author 猎魔人
 * @date 2024-10-24 从先前的直接法的tracker里抠出来的              
*/
#ifndef PHOTOMETRICFACTOR_H
#define PHOTOMETRICFACTOR_H
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include <boost/format.hpp>
#include <chrono>
#include <mutex>
//dependency for opencv
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
//dependency for eigen
#include <Eigen/Core>
#include <Eigen/Dense>

typedef Eigen::Matrix<float,  6, 6> Mat66f;
typedef Eigen::Matrix<double, 6, 6> Mat66d;
typedef Eigen::Matrix<float,  6, 1> Vec6f;
typedef Eigen::Matrix<double, 6, 1> Vec6d;

namespace eslam_core{

    float GetPixelValue(const cv::Mat &img, float x, float y);

    //-- 计算单层金字塔的雅可比矩阵与增量方程
	void accumulate_Jacbian_single_level(const cv::Mat& image_ref, const cv::Mat& image_cur, Sophus::SE3d transform,
                                         cv::Point queryPixel, float depth, 
                                         float fx, float fy, float cx, float cy,
                                         Mat66d& H_out, Vec6d& g_out, double& cost_out);

}//namespace dierct


#endif