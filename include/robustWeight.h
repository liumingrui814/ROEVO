#ifndef ROBUSTWEIGHT_H
#define ROBUSTWEIGHT_H

#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Core>

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <cmath>
#include <type_traits>

namespace eslam_core {

class rwUtils {
public:
    static float HuberWeight(const float& dist, const float k = 1.345f) 
    {
        if (dist <= k) {
            return 1.0f;      // Quadratic region
        } else {
            return k / dist;  // Linear region
        }
    }

    static float TukeyWeight(const float& dist, const float k = 4.685f)
    {
        const float abs_dist = std::abs(dist);
        if (abs_dist <= k) {
            const float r_over_k = dist / k;
            const float tmp = 1.0f - r_over_k * r_over_k;
            return tmp * tmp;  // (1 - (r/k)^2)^2
        } else {
            return 0.0f;  // Zero weight for outliers
        }
    }

    static float CauchyWeight(const float& dist, const float k = 2.3849f)
    {
        const float r_over_k = dist / k;
        return 1.0f / (1.0f + r_over_k * r_over_k);
    }
};

class robustWeight1D{
public:
    //-- 期望
    float expt;
    //-- 标准差
    float sigma;
    //-- 各个维度的残差
    std::vector<float> residuals;

    //-- 马氏距离
    std::vector<float> Mahalanobis;

    //-- 权重
    std::vector<float> weights;

    robustWeight1D(const std::vector<float>& input_vectors);

    void computeWeights(const std::string& type = "Huber" );

private:
    void computeStatistics();

    void computeMahalanobis();
};


class robustWeight2D{
public:
    //-- 期望
    Eigen::Vector2d expt;
    //-- 标准差
    Eigen::Vector2d sigma;
    //-- 各个维度的残差
    std::vector<Eigen::Vector2d> residuals;

    //-- 马氏距离
    std::vector<float> Mahalanobis;

    //-- 权重
    std::vector<float> weights;

    robustWeight2D(const std::vector<Eigen::Vector2d>& input_vectors);

    void computeWeights(const std::string& type = "Huber" );

private:
    void computeStatistics();

    void computeMahalanobis();

};

//-- 根据残差平方服从卡方分布的假设进行鲁棒权重设计
class robustWeightChi2
{
public:
    //-- 使用编译期常量定义 chi2 分布的 分位数
    static constexpr double CHI2_20_PERCENT = 1.642;  // 真正的类作用域常量
    static constexpr double CHI2_10_PERCENT = 2.706;
    static constexpr double CHI2_05_PERCENT = 3.841;
    static constexpr double CHI2_01_PERCENT = 6.635;
    
    double sigma2; //-- 残差平方的期望（也就是残差的方差）

    std::vector<double> residuals_2;
    std::vector<double> chi2residuals_2;
    std::vector<double> weight;
    
    robustWeightChi2(std::vector<double> errors_2)
    {
        residuals_2 = errors_2;
        computeStatistics();
    }

    void computeStatistics();

    double computeWight(double error);

    void computeWights();

};

class sigma_3_distribution
{
public:
    double sigma; //-- 残差构成的分布的标准差
    double expt; //-- 残差分布的期望
    double abs_med; //-- 残差分布的绝对值的中位数
    std::vector<double> residuals;
    sigma_3_distribution(std::vector<double> errors)
    {
        residuals = errors;
        constructDistribution();
    }

    void constructDistribution();

    double computeWight(double error);

};


//-- 支持任意维度的残差（nx1）
// template<int Dim>
// class robustWeightXD{
// public:
//     //-- 当类中包含固定大小的 Eigen 对象（如 Eigen::Vector2d、Eigen::Matrix4f 等）
//     //-- 作为成员变量，并且类会通过 new 动态分配时，必须使用此宏
//     EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

//     // 存储输入的向量
//     std::vector<Eigen::Vector<float, Dim, 1>> vectors;
    
//     // 向量的期望
//     Eigen::Vector<float, Dim, 1> vec_expt;
    
//     // 向量的标准差
//     Eigen::Vector<float, Dim, 1> vec_sigma;
    
//     // 改为非静态成员变量
//     const int dim = Dim;  // 或者直接使用 Dim，不需要额外存储

//     // 构造函数
//     robustWeightXD(const std::vector<Eigen::Vector<float, Dim, 1>>& input_vectors);

//     // 禁用错误类型的构造函数
//     template<typename T>
//     robustWeightXD(const std::vector<T>&) {
//         static_assert(std::is_same_v<T, Eigen::Vector<float, Dim, 1>>, 
//                      "Input must be std::vector<Eigen::Vector<float, Dim, 1>>");
//     }

// private:
//     void computeStatistics();

// };


}

#endif // ROBUSTWEIGHT_H
