#include "directTracker.h"
#include <chrono>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>

// #define __VISUALIZE_PHOTOMETRIC__

using namespace direct;

int patternNum = 12;
int pattern[12][2] = {{0,-2}, {0,2}, {-2,0}, {2,0}, {0,1}, {1,1}, {1,0},{1,-1},{0,-1}, {-1,-1},{-1,0},{-1,1}};
int normalPatternNum = 8;
int normalPattern[8][2] = {{0,1}, {1,1}, {1,0},{1,-1},{0,-1}, {-1,-1},{-1,0},{-1,1}};
int briefPatternNum = 4;
int briefPattern[4][2] = {{0,1}, {1,0}, {-1,0}, {0,-1}};

bool isPointInVector(const cv::Point& point, const std::vector<directPoint>& pointVector)
{  
    for (const auto& p : pointVector){
        if (cvRound(p.x_2d) == cvRound(point.x) && cvRound(p.x_2d) == cvRound(point.y))
            return true;
    }
    return false;
}

Eigen::Matrix2d theta2R(const float& theta)
{
    Eigen::Matrix2d R;
    double cos_theta = std::cos(theta);
    double sin_theta = std::sin(theta);
    // 旋转矩阵公式:
    // [ cosθ  -sinθ ]
    // [ sinθ   cosθ ]
    R << cos_theta, -sin_theta,
         sin_theta,  cos_theta;

    return R;
}

//-- 初始化本类
DirectTracker::DirectTracker(int w, int h, double fx, double fy, double cx, double cy)
{
    assignPyramid(w, h, fx, fy, cx, cy);
}

//-- 设置参考帧的图像金字塔与特征点金字塔
void DirectTracker::setReference(const cv::Mat& reference_image, 
	                  std::vector<float>& x_list, std::vector<float>& y_list,
					  std::vector<float>& depth_list, std::vector<float>& weight_list,
					  std::vector<float>& theta_list)
{
    //-- input image has to be gray scale
    assert(reference_image.channels() == 1);
    //-- 输入参数的列表需要大小一致
    int N = x_list.size();
    assert(N == y_list.size() && N == depth_list.size() && N == weight_list.size());

    mvPyrImagesRef.clear();
    mvPyrPoints2D.clear();



    //-- 计算参考帧的金字塔
    for (size_t i = 0; i < PYR_LEVELS; i++) 
    {
        if (i == 0) 
        {
            mvPyrImagesRef.push_back(reference_image);
        }else{
            cv::Mat img_pyr;
            cv::resize(mvPyrImagesRef[i - 1], img_pyr, cv::Size(mvWidth[i], mvHeight[i]));
            mvPyrImagesRef.push_back(img_pyr);
        }
    }

    //-- 构造参与直接法的图像点
    std::vector<directPoint> bottom_pts(N);
    for(int i = 0; i < N; ++i)
    {
        bottom_pts[i] = directPoint(x_list[i], y_list[i], depth_list[i], weight_list[i], theta_list[i]);
    }

    mvPyrPoints2D.push_back(bottom_pts);

    //-- 构建直接点的金字塔，移除缩放后重合的点
    for(int i = 1; i < PYR_LEVELS; ++i)
    {
        std::vector<directPoint> layer_pts;
        for(const auto &point : mvPyrPoints2D[i-1])
        {  
            int newX = cvRound(float(point.x_2d) * PYR_SCALE);  
            int newY = cvRound(float(point.y_2d) * PYR_SCALE);
            //-- 只插入不重复的点 
            if(!isPointInVector(cv::Point(newX, newY), layer_pts))
            {
                directPoint pt = point;
                pt.x_2d = newX;
                pt.y_2d = newY;
                layer_pts.push_back(pt);
            }
        }
        mvPyrPoints2D.push_back(layer_pts);
    }
}

//-- 设置当前帧的图像金字塔以及图像的梯度矩阵
void DirectTracker::setCurrent(cv::Mat current_image)
{
    assert(current_image.channels() == 1);
    mvPyrImagesCur.clear();

    for (int i = 0; i < PYR_LEVELS; i++) 
    {
        if (i == 0) {
            mvPyrImagesCur.push_back(current_image);
        } else {
            cv::Mat img_pyr;
            cv::resize(mvPyrImagesCur[i-1], img_pyr, cv::Size(mvWidth[i], mvHeight[i]));
            mvPyrImagesCur.push_back(img_pyr);
        }
    }
}

//-- 得到多层金字塔分别对应的相机参数
void DirectTracker::assignPyramid(int w, int h, double fx, double fy, double cx, double cy)
{
    //-- 由于图像金字塔的参数是共享的，因此作为本类的共有变量统一处理
    mvWidth[0] = w;
    mvHeight[0] = h;
    
    //-- 首层内参赋值
	mvFx[0] = fx; mvCx[0] = cx;
	mvFy[0] = fy; mvCy[0] = cy;

	//-- 内参与图像大小的金字塔层数赋值
	for (int level = 1; level < PYR_LEVELS; ++ level)
	{
		mvWidth[level] = int(mvWidth[level-1] * PYR_SCALE);
		mvHeight[level] = int(mvHeight[level-1] * PYR_SCALE);
		mvFx[level] = mvFx[level-1] * PYR_SCALE;
		mvFy[level] = mvFy[level-1] * PYR_SCALE;
		mvCx[level] = mvCx[level-1] * PYR_SCALE;
		mvCy[level] = mvCx[level-1] * PYR_SCALE;
        mvK[level]  << mvFx[level], 0.0, mvCx[level], 0.0, mvFy[level], mvCy[level], 0.0, 0.0, 1.0;
	}
}

//-- 使用双线性插值得到图片在浮点坐标下的像素值
float DirectTracker::GetPixelValue(const cv::Mat &img, float x, float y) 
{
    //-- 边缘检查（？x,y超出图像边缘按图像边缘赋值）
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= img.cols) x = img.cols - 1;
    if (y >= img.rows) y = img.rows - 1;
    //-- 灰度图像的线性指针
    int step = (int)img.step1();
    uchar *data = &img.data[int(y) * step + int(x)];
    float xx = x - floor(x); //-- 双线性插值需要的横向比例
    float yy = y - floor(y); //-- 双线性插值需要的纵向比例
    //-- 利用data[0]，data[1],以及data[x],data[x+1]进行双线性插值
    return float(
        (1 - xx) * (1 - yy) * data[0] +
        xx * (1 - yy) * data[1] +
        (1 - xx) * yy * data[step] +
        xx * yy * data[step + 1]
    );
}

//-- 计算单层金字塔的雅可比矩阵
void DirectTracker::accumulate_Jacbian_single_level(int lvl, Sophus::SE3d transform, double& cost_out,
										          normalLeastSquares& ls)
{
    //-- 对应金字塔的图像大小
    int wl = mvWidth[lvl], hl = mvHeight[lvl];
	//-- 当前帧图像
    cv::Mat image_cur = mvPyrImagesCur[lvl];
    //-- 参考帧图像
    cv::Mat image_ref = mvPyrImagesRef[lvl];
	//-- 平移
	Eigen::Vector3f t = transform.translation().cast<float>();

	//-- 相机参数
	float fxl = mvFx[lvl];
	float fyl = mvFy[lvl];
	float cxl = mvCx[lvl];
	float cyl = mvCy[lvl];

    std::vector<directPoint> point_set = mvPyrPoints2D[lvl];

    double cost_tmp = 0;
    //-- 光度误差并不是基于单个像素，而是基于整个块, 块大小即half_patch_size
    const int half_patch_size = 0;
    int cnt_good = 0;

    std::vector<Mat66d> hessian_list;
    std::vector<Vec6d> jacobian_list;
    std::vector<double> residuals; 

    for(std::size_t i = 0; i < point_set.size(); ++i){
        
        double weight = point_set[i].weight; //-- 观测噪声带来的深度不确定性

        //-- 使用内参反投影得到参考帧下归一化的相机坐标点
        float ref_norm_x = (point_set[i].x_2d - cxl)/fxl;
        float ref_norm_y = (point_set[i].y_2d - cyl)/fyl;
        Eigen::Vector3d point_ref = Eigen::Vector3d(ref_norm_x, ref_norm_y, 1);
        //-- 使用先验深度得到参考帧坐标系下实际的3D点
        point_ref *= point_set[i].depth;

        //-- 利用位姿变换得到当前帧下点的3D坐标
        Eigen::Vector3d point_cur = transform * point_ref;
        if(point_cur[2] < 0) continue;//-- 深度为负的点不考虑放入优化模型中

        //-- 利用内参投影得到点在当前帧的像素坐标
        float u = fxl * point_cur[0] / point_cur[2] + cxl;
        float v = fyl * point_cur[1] / point_cur[2] + cyl;
        //-- 投影出图像的点不考虑放入优化模型中
        if (u < half_patch_size || u > wl - half_patch_size || 
            v < half_patch_size || v > hl - half_patch_size) continue;

        //-- 能走到这一步的点都是有效点
        cnt_good++;

        Eigen::Vector2d projectPoint = Eigen::Vector2d(u,v);
        
        double X = point_cur[0], Y = point_cur[1], Z = point_cur[2];
        double Z2 = Z * Z, Z_inv = 1.0 / Z, Z2_inv = Z_inv * Z_inv;
        

        //-- 像素关于李代数的雅克比矩阵，维度为2x6
        Eigen::Matrix<double, 2, 6> J_pixel_xi;
        //-- 具体推导可看slam14讲或是本目录下文档的推导
        J_pixel_xi(0, 0) = fxl * Z_inv;
        J_pixel_xi(0, 1) = 0;
        J_pixel_xi(0, 2) = -fxl * X * Z2_inv;
        J_pixel_xi(0, 3) = -fxl * X * Y * Z2_inv;
        J_pixel_xi(0, 4) = fxl + fxl * X * X * Z2_inv;
        J_pixel_xi(0, 5) = -fxl * Y * Z_inv;

        J_pixel_xi(1, 0) = 0;
        J_pixel_xi(1, 1) = fyl * Z_inv;
        J_pixel_xi(1, 2) = -fyl * Y * Z2_inv;
        J_pixel_xi(1, 3) = -fyl - fyl * Y * Y * Z2_inv;
        J_pixel_xi(1, 4) = fyl * X * Y * Z2_inv;
        J_pixel_xi(1, 5) = fyl * X * Z_inv;

        //-- 当前单个点的雅克比矩阵和嗨森矩阵，用于迭代更新LS求解器模型
        Mat66d H_bb_curr = Mat66d::Zero();
        Vec6d  g_b_curr  =  Vec6d::Zero();
        double cost_curr = 0;


        //-- 基于块的光度误差更新雅克比矩阵
        for (int x = -half_patch_size; x <= half_patch_size; x++){
            for (int y = -half_patch_size; y <= half_patch_size; y++) {
                //-- point_set[i]是参考帧的位置，(u,v) 是当前帧的位置，按块来计算光度误差
                double error = GetPixelValue(image_ref, point_set[i].x_2d + x, point_set[i].y_2d + y) -
                               GetPixelValue(image_cur, u + x, v + y);

                //-- 图像梯度的雅克比矩阵
                Eigen::Vector2d J_img_pixel;

                //-- 图像梯度的计算，根据原理是计算当前帧的图像梯度
                J_img_pixel = Eigen::Vector2d(
                    0.5 * (GetPixelValue(image_cur, u + 1 + x, v + y) - GetPixelValue(image_cur, u - 1 + x, v + y)),
                    0.5 * (GetPixelValue(image_cur, u + x, v + 1 + y) - GetPixelValue(image_cur, u + x, v - 1 + y))
                );

                //-- 整体的Jacobi = dI/dp * dp/d\xi, 这里是负数是因为代码中残差error是I1-I2
                Vec6d J = -1.0 * (J_img_pixel.transpose() * J_pixel_xi).transpose();

                H_bb_curr += J * J.transpose() * weight * weight;
                g_b_curr += -error * J * weight * weight;
                cost_curr += error * error;
            }
        }

        hessian_list.push_back(H_bb_curr);
        jacobian_list.push_back(g_b_curr);
        residuals.push_back(cost_curr);
    }
    //-- 得到残差列表、雅克比列表后，根据残差的t分布构建最小二乘问题
    eslam_core::sigma_3_distribution dst(residuals);

    normalLeastSquares ls_temp;
    for(int i = 0; i < jacobian_list.size(); ++i){
        ls_temp.updateBlock(hessian_list[i], jacobian_list[i], residuals[i], dst.computeWight(residuals[i]));
    }
    ls_temp.finish();
    ls = ls_temp;

    cost_out = cost_tmp;
}

//-- 并行计算单层金字塔的雅可比矩阵
void DirectTracker::accumulate_Jacbian_single_level_Parallel(int lvl, Sophus::SE3d transform, double& cost_out,
										          normalLeastSquares& ls)
{
    //-- 对应金字塔的图像大小
    int wl = mvWidth[lvl], hl = mvHeight[lvl];
	//-- 当前帧图像
    cv::Mat image_cur = mvPyrImagesCur[lvl];
    //-- 参考帧图像
    cv::Mat image_ref = mvPyrImagesRef[lvl];
	//-- 平移
	Eigen::Vector3f t = transform.translation().cast<float>();

	//-- 相机参数
	float fxl = mvFx[lvl];
	float fyl = mvFy[lvl];
	float cxl = mvCx[lvl];
	float cyl = mvCy[lvl];

    std::vector<directPoint> point_set = mvPyrPoints2D[lvl];

    double cost_tmp = 0;
    //-- 光度误差并不是基于单个像素，而是基于整个块, 块大小即half_patch_size
    const int half_patch_size = 1;

    std::vector<Mat66d> hessian_list;
    std::vector<Vec6d> jacobian_list;
    std::vector<double> residuals; 

    //-- 为了并行计算，提前分配好存储空间
    hessian_list.resize(point_set.size());
    jacobian_list.resize(point_set.size());
    residuals.resize(point_set.size());

    //-- 原子索引
    std::atomic<size_t> actualSize(0);

    tbb::parallel_for(tbb::blocked_range<size_t>(0, point_set.size()),
        [&](const tbb::blocked_range<size_t>& range){
            for (size_t i = range.begin(); i != range.end(); ++i) {
                //-- 观测噪声带来的深度不确定性
                double weight = point_set[i].weight; 

                //-- 使用内参反投影得到参考帧下归一化的相机坐标点
                float ref_norm_x = (point_set[i].x_2d - cxl)/fxl;
                float ref_norm_y = (point_set[i].y_2d - cyl)/fyl;
                Eigen::Vector3d point_ref = Eigen::Vector3d(ref_norm_x, ref_norm_y, 1);
                //-- 使用先验深度得到参考帧坐标系下实际的3D点
                point_ref *= point_set[i].depth;

                //-- 利用位姿变换得到当前帧下点的3D坐标
                Eigen::Vector3d point_cur = transform * point_ref;
                if(point_cur[2] < 0) continue;//-- 深度为负的点不考虑放入优化模型中

                //-- 利用内参投影得到点在当前帧的像素坐标
                float u = fxl * point_cur[0] / point_cur[2] + cxl;
                float v = fyl * point_cur[1] / point_cur[2] + cyl;
                //-- 投影出图像的点不考虑放入优化模型中
                if (u < half_patch_size || u > wl - half_patch_size || 
                    v < half_patch_size || v > hl - half_patch_size) continue;

                Eigen::Vector2d projectPoint = Eigen::Vector2d(u,v);
                
                double X = point_cur[0], Y = point_cur[1], Z = point_cur[2];
                double Z2 = Z * Z, Z_inv = 1.0 / Z, Z2_inv = Z_inv * Z_inv;
                

                //-- 像素关于李代数的雅克比矩阵，维度为2x6
                Eigen::Matrix<double, 2, 6> J_pixel_xi;
                //-- 具体推导可看slam14讲或是本目录下文档的推导
                J_pixel_xi(0, 0) = fxl * Z_inv;
                J_pixel_xi(0, 1) = 0;
                J_pixel_xi(0, 2) = -fxl * X * Z2_inv;
                J_pixel_xi(0, 3) = -fxl * X * Y * Z2_inv;
                J_pixel_xi(0, 4) = fxl + fxl * X * X * Z2_inv;
                J_pixel_xi(0, 5) = -fxl * Y * Z_inv;

                J_pixel_xi(1, 0) = 0;
                J_pixel_xi(1, 1) = fyl * Z_inv;
                J_pixel_xi(1, 2) = -fyl * Y * Z2_inv;
                J_pixel_xi(1, 3) = -fyl - fyl * Y * Y * Z2_inv;
                J_pixel_xi(1, 4) = fyl * X * Y * Z2_inv;
                J_pixel_xi(1, 5) = fyl * X * Z_inv;

                //-- 当前单个点的雅克比矩阵和嗨森矩阵，用于迭代更新LS求解器模型
                Mat66d H_bb_curr = Mat66d::Zero();
                Vec6d  g_b_curr  =  Vec6d::Zero();
                double cost_curr = 0;


                //-- 基于块的光度误差更新雅克比矩阵
                for (int x = -half_patch_size; x <= half_patch_size; x++){
                    for (int y = -half_patch_size; y <= half_patch_size; y++) {
                        //-- point_set[i]是参考帧的位置，(u,v) 是当前帧的位置，按块来计算光度误差
                        double error = GetPixelValue(image_ref, point_set[i].x_2d + x, point_set[i].y_2d + y) -
                                    GetPixelValue(image_cur, u + x, v + y);

                        //-- 图像梯度的雅克比矩阵
                        Eigen::Vector2d J_img_pixel;

                        //-- 图像梯度的计算，根据原理是计算当前帧的图像梯度
                        J_img_pixel = Eigen::Vector2d(
                            0.5 * (GetPixelValue(image_cur, u + 1 + x, v + y) - GetPixelValue(image_cur, u - 1 + x, v + y)),
                            0.5 * (GetPixelValue(image_cur, u + x, v + 1 + y) - GetPixelValue(image_cur, u + x, v - 1 + y))
                        );

                        //-- 整体的Jacobi = dI/dp * dp/d\xi, 这里是负数是因为代码中残差error是I1-I2
                        Vec6d J = -1.0 * (J_img_pixel.transpose() * J_pixel_xi).transpose();

                        H_bb_curr += J * J.transpose() * weight * weight;
                        g_b_curr += -error * J * weight * weight;
                        cost_curr += error * error;
                    }
                }
                
                size_t pos = actualSize.fetch_add(1);
                
                hessian_list[pos] = std::move(H_bb_curr);
                jacobian_list[pos] = std::move(g_b_curr);
                residuals[pos] = std::move(cost_curr);
                
            }
        });

    hessian_list.resize(actualSize);
    jacobian_list.resize(actualSize);
    residuals.resize(actualSize);

    eslam_core::sigma_3_distribution dst(residuals);

    normalLeastSquares ls_temp;
    for(int i = 0; i < jacobian_list.size(); ++i){
        ls_temp.updateBlock(hessian_list[i], jacobian_list[i], residuals[i], dst.computeWight(residuals[i]));
    }
    ls_temp.finish();
    ls = ls_temp;

    cost_out = cost_tmp;
}

void DirectTracker::accumulate_Jacbian_single_level_RI(int lvl, Sophus::SE3d transform, double& cost_out,
    normalLeastSquares& ls)
{
    //-- 对应金字塔的图像大小
    int wl = mvWidth[lvl], hl = mvHeight[lvl];
	//-- 当前帧图像
    cv::Mat image_cur = mvPyrImagesCur[lvl];
    //-- 参考帧图像
    cv::Mat image_ref = mvPyrImagesRef[lvl];
	//-- 平移
	Eigen::Vector3f t = transform.translation().cast<float>();

	//-- 相机参数
	float fxl = mvFx[lvl];
	float fyl = mvFy[lvl];
	float cxl = mvCx[lvl];
	float cyl = mvCy[lvl];

    std::vector<directPoint> point_set = mvPyrPoints2D[lvl];

    // 计算图像的x和y方向梯度（Scharr 算子）
    cv::Mat gradX, gradY;
    cv::Scharr(image_cur, gradX, CV_32F, 1, 0); // dx (水平梯度)
    cv::Scharr(image_cur, gradY, CV_32F, 0, 1); // dy (垂直梯度)

    double cost_tmp = 0;

    //-- 光度误差并不是基于单个像素，而是基于整个块, 块大小即half_patch_size
    //-- 2 是 pattern[12][2] 的邻域的边界 bias
    const int half_patch_size = 2;

    std::vector<Mat66d> hessian_list;
    std::vector<Vec6d> jacobian_list;
    std::vector<double> residuals; 

    //-- 为了并行计算，提前分配好存储空间
    hessian_list.resize(point_set.size());
    jacobian_list.resize(point_set.size());
    residuals.resize(point_set.size());

    //-- 原子索引
    std::atomic<size_t> actualSize(0);

    tbb::parallel_for(tbb::blocked_range<size_t>(0, point_set.size()),
        [&](const tbb::blocked_range<size_t>& range){
            for (size_t i = range.begin(); i != range.end(); ++i) {
                //-- 观测噪声带来的深度不确定性
                double weight = point_set[i].weight; 

                //-- 使用内参反投影得到参考帧下归一化的相机坐标点
                float ref_norm_x = (point_set[i].x_2d - cxl)/fxl;
                float ref_norm_y = (point_set[i].y_2d - cyl)/fyl;
                Eigen::Vector3d point_ref = Eigen::Vector3d(ref_norm_x, ref_norm_y, 1);
                //-- 使用先验深度得到参考帧坐标系下实际的3D点
                point_ref *= point_set[i].depth;

                //-- 利用位姿变换得到当前帧下点的3D坐标
                Eigen::Vector3d point_cur = transform * point_ref;
                if(point_cur[2] < 0) continue;//-- 深度为负的点不考虑放入优化模型中

                //-- 利用内参投影得到点在当前帧的像素坐标
                float u = fxl * point_cur[0] / point_cur[2] + cxl;
                float v = fyl * point_cur[1] / point_cur[2] + cyl;
                //-- 投影出图像的点不考虑放入优化模型中
                if (u < half_patch_size || u > wl - half_patch_size || 
                    v < half_patch_size || v > hl - half_patch_size) continue;

                Eigen::Vector2d projectPoint = Eigen::Vector2d(u,v);
                
                double X = point_cur[0], Y = point_cur[1], Z = point_cur[2];
                double Z2 = Z * Z, Z_inv = 1.0 / Z, Z2_inv = Z_inv * Z_inv;
                

                //-- 像素关于李代数的雅克比矩阵，维度为2x6
                Eigen::Matrix<double, 2, 6> J_pixel_xi;
                //-- 具体推导可看slam14讲或是本目录下文档的推导
                J_pixel_xi(0, 0) = fxl * Z_inv;
                J_pixel_xi(0, 1) = 0;
                J_pixel_xi(0, 2) = -fxl * X * Z2_inv;
                J_pixel_xi(0, 3) = -fxl * X * Y * Z2_inv;
                J_pixel_xi(0, 4) = fxl + fxl * X * X * Z2_inv;
                J_pixel_xi(0, 5) = -fxl * Y * Z_inv;

                J_pixel_xi(1, 0) = 0;
                J_pixel_xi(1, 1) = fyl * Z_inv;
                J_pixel_xi(1, 2) = -fyl * Y * Z2_inv;
                J_pixel_xi(1, 3) = -fyl - fyl * Y * Y * Z2_inv;
                J_pixel_xi(1, 4) = fyl * X * Y * Z2_inv;
                J_pixel_xi(1, 5) = fyl * X * Z_inv;

                //-- 当前单个点的雅克比矩阵和嗨森矩阵，用于迭代更新LS求解器模型
                Mat66d H_bb_curr = Mat66d::Zero();
                Vec6d  g_b_curr  =  Vec6d::Zero();
                double cost_curr = 0;


                //-- 基于块的光度误差更新雅克比矩阵
                for (size_t k = 0; k < patternNum; ++k)
                {
                    //-- patch 的坐标
                    int x = pattern[k][0];
                    int y = pattern[k][1];
                    Eigen::Vector2d patch_bias;
                    patch_bias << double(x),double(y);

                    //-- I1 中 patch 的对应坐标
                    float theta_1 = point_set[i].theta;
                    Eigen::Matrix2d R1 = theta2R(theta_1);
                    Eigen::Vector2d ref_bias = patch_bias;

                    //-- I2 中 patch 的对应坐标
                    // 获取当前点的梯度分量
                    float dx = gradX.at<float>(cvRound(v), cvRound(u));
                    float dy = gradY.at<float>(cvRound(v), cvRound(u));
                    float theta_2 = std::atan2(dy, dx);
                    Eigen::Matrix2d R2 = theta2R(theta_2);
                    Eigen::Vector2d curr_bias = R2 * R1.inverse() * patch_bias;

                    //-- point_set[i]是参考帧的位置，(u,v) 是当前帧的位置，按块来计算光度误差
                    //-- 由 rotational invariant 的 patch 对来计算光度误差
                    double error = GetPixelValue(image_ref, point_set[i].x_2d + ref_bias.x(), point_set[i].y_2d + ref_bias.y()) -
                                   GetPixelValue(image_cur, u + curr_bias.x(), v + curr_bias.y());

                    //-- 图像梯度的雅克比矩阵
                    Eigen::Vector2d J_img_pixel;

                    //-- 图像梯度的计算，根据原理是计算当前帧的图像梯度
                    J_img_pixel = Eigen::Vector2d(
                        0.5 * (GetPixelValue(image_cur, u + 1 + curr_bias.x(), v + curr_bias.y()) - GetPixelValue(image_cur, u - 1 + curr_bias.x(), v + curr_bias.y())),
                        0.5 * (GetPixelValue(image_cur, u + curr_bias.x(), v + 1 + curr_bias.y()) - GetPixelValue(image_cur, u + curr_bias.x(), v - 1 + curr_bias.y()))
                    );

                    //-- 整体的Jacobi = dI/dp * dp/d\xi, 这里是负数是因为代码中残差error是I1-I2
                    Vec6d J = -1.0 * (J_img_pixel.transpose() * J_pixel_xi).transpose();

                    H_bb_curr += J * J.transpose() * weight * weight;
                    g_b_curr += -error * J * weight * weight;
                    cost_curr += error * error;
                }
                
                size_t pos = actualSize.fetch_add(1);
                
                hessian_list[pos] = std::move(H_bb_curr);
                jacobian_list[pos] = std::move(g_b_curr);
                residuals[pos] = std::move(cost_curr);
                
            }
        });

    hessian_list.resize(actualSize);
    jacobian_list.resize(actualSize);
    residuals.resize(actualSize);
    
    //-- 根据残差配置鲁棒权重以解最小二乘问题
    eslam_core::sigma_3_distribution dst(residuals);

    normalLeastSquares ls_temp;
    for(int i = 0; i < jacobian_list.size(); ++i){
        ls_temp.updateBlock(hessian_list[i], jacobian_list[i], residuals[i], dst.computeWight(residuals[i]));
    }
    ls_temp.finish();
    ls = ls_temp;

    cost_out = cost_tmp;
}

//-- 使用高斯牛顿法优化单层金字塔
void DirectTracker::estimateSingleLayer(int lvl, Sophus::SE3d& T21, bool use_parallel, bool use_rotational_invariant)
{
    const int iterations = 100;
    double cost = 0, lastCost = FLT_MAX;

    for (int iter = 0; iter < iterations; iter++) {
        double cost;
        normalLeastSquares ls;
        if(use_parallel == true && use_rotational_invariant == false)
        {
            accumulate_Jacbian_single_level_Parallel(lvl, T21, cost, ls);
        }else if (use_parallel == false && use_rotational_invariant == false){
            accumulate_Jacbian_single_level(lvl, T21, cost, ls);
        }else{
            accumulate_Jacbian_single_level_RI(lvl, T21, cost, ls);
        }

        Vec6d update;
        ls.slove(update);
        T21 = Sophus::SE3d::exp(update) * T21;

        if (std::isnan(update[0])) {
            std::cout << "\033[31m [ERROR][DIRECT] \033[0m"<<": update is nan!!! \n";
            break;
        }
        if (iter > 0 && cost > lastCost) break;
        if (update.norm() < 1e-3) break;

        lastCost = cost;
    }
}

//-- 多层直接法估计位姿变换，遍历整个金字塔得到结果
void DirectTracker::estimatePyramid(Sophus::SE3d &T21, bool use_parallel)
{
    for (int level = PYR_LEVELS - 1; level >= 0; level--) {
        estimateSingleLayer(level, T21, use_parallel, false);
    }
    estimateSingleLayer(0, T21, use_parallel, true);

}



