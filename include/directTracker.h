#ifndef DIRECTTRACKER_H
#define DIRECTTRACKER_H
#include <boost/format.hpp>
#include <chrono>
#include <mutex>
#include <iostream>
//dependency for opencv
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
//dependency for eigen
#include <Eigen/Core>
#include <Eigen/Dense>
//dependency for sophus
#include <sophus/se3.hpp>
//dependency for tbb multi-thread
#include <tbb/tbb.h>
#include <tbb/parallel_for.h>
//least square solver
#include "Least_squares.h"
#include "robustWeight.h"

#define PYR_LEVELS 5
#define PYR_SCALE 0.5
#define NAN FLT_MAX

namespace direct{

typedef Eigen::Matrix<float,  6, 6> Mat66f;
typedef Eigen::Matrix<double, 6, 6> Mat66d;
typedef Eigen::Matrix<float,  6, 1> Vec6f;
typedef Eigen::Matrix<double, 6, 1> Vec6d;
typedef std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;

// extern int patternNum;
// extern int pattern[12][2];
// extern int normalPatternNum;
// extern int normalPattern[8][2];
// extern int briefPatternNum;
// extern int briefPattern[4][2];

class directPoint{
public:
	//-- 点的像素坐标
	float x_2d;
	float y_2d;
	//-- 点的正深度
	float depth;
	//-- 点在优化中的初始权重
	float weight;
	//-- 点的梯度方向(-pi -- pi)
	float theta;

	//-- 构造函数
	directPoint(){}
	directPoint(float _x_2d, float _y_2d, float _depth, float _weight, float _theta)
	:x_2d(_x_2d), y_2d(_y_2d), depth(_depth), weight(_weight),theta(_theta){}
};

class DirectTracker {
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

	DirectTracker(int w, int h, double fx, double fy, double cx, double cy);

	//-- 设置参考帧，参考帧包括图像金字塔与选点, 以及为每个选点赋值先验深度
	void setReference(const cv::Mat& reference_image, 
	                  std::vector<float>& x_list, std::vector<float>& y_list,
					  std::vector<float>& depth_list, std::vector<float>& weight_list,
					  std::vector<float>& theta_list);
	//-- 设置当前帧，当前帧包括图像金字塔与图像梯度
	void setCurrent(cv::Mat current_image);

	void estimatePyramid(Sophus::SE3d &T21, bool use_parallel = true);

private:
	Eigen::Matrix3d mvK[PYR_LEVELS];  //-- 成员变量，内参金字塔
	double mvFx[PYR_LEVELS];          //-- 成员变量，fx的金字塔
	double mvFy[PYR_LEVELS];          //-- 成员变量，fy的金字塔
	double mvCx[PYR_LEVELS];          //-- 成员变量，cx的金字塔
	double mvCy[PYR_LEVELS];          //-- 成员变量，cy的金字塔
	int mvWidth[PYR_LEVELS];          //-- 成员变量，图像宽金字塔
	int mvHeight[PYR_LEVELS];         //-- 成员变量，图像高金字塔

	//-- 参考帧金字塔的特征点位置
	std::vector<std::vector<directPoint>> mvPyrPoints2D;
	//-- 参考帧图像金字塔
	std::vector<cv::Mat> mvPyrImagesRef;
	//-- 当前帧图像金字塔
	std::vector<cv::Mat> mvPyrImagesCur;


	
	//-- 初始化图像金字塔参数
	void assignPyramid(int w, int h, double fx, double fy, double cx, double cy);

	//-- 使用双线性插值获取图像在某个小数坐标位置的像素值
	float GetPixelValue(const cv::Mat &img, float x, float y);

	//-- 计算单层金字塔的雅可比矩阵与增量方程
	void accumulate_Jacbian_single_level(int lvl, Sophus::SE3d transform, double& cost_out,
										 normalLeastSquares& ls);
	
	void accumulate_Jacbian_single_level_Parallel(int lvl, Sophus::SE3d transform, double& cost_out,
										normalLeastSquares& ls);

	void accumulate_Jacbian_single_level_RI(int lvl, Sophus::SE3d transform, double& cost_out,
		                                    normalLeastSquares& ls);

	//-- 使用高斯牛顿法优化单层金字塔
	void estimateSingleLayer(int lvl, Sophus::SE3d& T21, bool use_parallel, bool use_rotational_invariant = false);
};

}//namespace dierct


#endif