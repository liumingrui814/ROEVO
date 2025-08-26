#include "photometricFactor.h"
using namespace eslam_core;

float eslam_core::GetPixelValue(const cv::Mat &img, float x, float y)
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

//-- 注意，与几何残差不同的是光度残差的点并不能出图像边界，几何残差是可以出图像边界的
void eslam_core::accumulate_Jacbian_single_level(const cv::Mat& image_ref, const cv::Mat& image_cur, Sophus::SE3d transform,
                                         cv::Point queryPixel, float depth,
                                         float fx, float fy, float cx, float cy,
                                         Mat66d& H_out, Vec6d& g_out, double& cost_out)
{
    //-- 重置所有的矩阵的值，这样返回无效值就都是0
    Mat66d H_bb_curr = Mat66d::Zero(); H_out = H_bb_curr;
    Vec6d  g_b_curr  =  Vec6d::Zero(); g_out = g_b_curr;
    cost_out = 0;

    //-- 光度误差并不是基于单个像素，而是基于整个块, 块大小即half_patch_size
    const int half_patch_size = 2;
    int wl = image_ref.cols, hl = image_ref.rows;

    //-- 使用内参反投影得到参考帧下归一化的相机坐标点
    float ref_norm_x = (queryPixel.x - cx)/fx;
    float ref_norm_y = (queryPixel.y - cy)/fy;
    Eigen::Vector3d point_ref = Eigen::Vector3d(ref_norm_x, ref_norm_y, 1);
    //-- 使用先验深度得到参考帧坐标系下实际的3D点
    point_ref *= depth;
    //-- 利用位姿变换得到当前帧下点的3D坐标
    Eigen::Vector3d point_cur = transform * point_ref;
    if(point_cur.z() < 0) return;//-- 深度为负的点不考虑放入优化模型中

    //-- 利用内参投影得到点在当前帧的像素坐标
    float u = fx * point_cur.x() / point_cur.z() + cx;
    float v = fy * point_cur.y() / point_cur.z() + cy;
    //-- 投影出图像的点不考虑放入优化模型中
    if (u < half_patch_size || u > wl - half_patch_size || 
        v < half_patch_size || v > hl - half_patch_size) return;
    
    double X = point_cur.x(), Y = point_cur.y(), Z = point_cur.z();
    double Z2 = Z * Z, Z_inv = 1.0 / Z, Z2_inv = Z_inv * Z_inv;

    //-- 像素关于李代数的雅克比矩阵，维度为2x6
    Eigen::Matrix<double, 2, 6> J_pixel_xi;
    //-- 具体推导可看slam14讲或是本目录下文档的推导
    J_pixel_xi(0, 0) = fx * Z_inv;
    J_pixel_xi(0, 1) = 0;
    J_pixel_xi(0, 2) = -fx * X * Z2_inv;
    J_pixel_xi(0, 3) = -fx * X * Y * Z2_inv;
    J_pixel_xi(0, 4) = fx + fx * X * X * Z2_inv;
    J_pixel_xi(0, 5) = -fx * Y * Z_inv;

    J_pixel_xi(1, 0) = 0;
    J_pixel_xi(1, 1) = fy * Z_inv;
    J_pixel_xi(1, 2) = -fy * Y * Z2_inv;
    J_pixel_xi(1, 3) = -fy - fy * Y * Y * Z2_inv;
    J_pixel_xi(1, 4) = fy * X * Y * Z2_inv;
    J_pixel_xi(1, 5) = fy * X * Z_inv;

    //-- 基于块的光度误差更新雅克比矩阵
    for (int x = -half_patch_size; x <= half_patch_size; x++){
        for (int y = -half_patch_size; y <= half_patch_size; y++) {
            //-- point_set[i]是参考帧的位置，(u,v) 是当前帧的位置，按块来计算光度误差
            double error = GetPixelValue(image_cur, queryPixel.x + x, queryPixel.y + y) -
                           GetPixelValue(image_ref, u + x, v + y);

            //-- 图像梯度的雅克比矩阵
            Eigen::Vector2d J_img_pixel;

            //-- 图像梯度的计算，根据原理是计算当前帧的图像梯度
            J_img_pixel = Eigen::Vector2d(
                0.5 * (GetPixelValue(image_ref, u + 1 + x, v + y) - GetPixelValue(image_ref, u - 1 + x, v + y)),
                0.5 * (GetPixelValue(image_ref, u + x, v + 1 + y) - GetPixelValue(image_ref, u + x, v - 1 + y))
            );

            //-- 整体的Jacobi = dI/dp * dp/d\xi, 这里是负数是因为代码中残差error是I1-I2
            Vec6d J = -1.0 * (J_img_pixel.transpose() * J_pixel_xi).transpose();

            H_out += J * J.transpose();
            g_out += -error * J;
            cost_out += error * error;
            //cost_out += error;
        }
    }
}
