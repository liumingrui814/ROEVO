#ifndef CALWEIGHT_H
#define CALWEIGHT_H

#include <Eigen/Geometry>
#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <vector>
#include <map>

namespace dvo_core {

class Tdistribution
{
public:
    float dof_;   // degree of freedom, can find in the paper
    float inital_sigma_;

    Tdistribution():dof_(5.0f),inital_sigma_(5.0)
    {

    }
    void configparam(float x)
    {
        dof_ = x;
    }

    float TdistributionScaleEsitimator(std::vector<double> errors);
    float WeightVaule(const float& x);

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

class WeightCaculation
{
public:
    WeightCaculation():sigma_(1.0f){}
    void calculateScale(std::vector<double> residuals);
    void computeWights(std::vector<double> residuals, std::vector<double>& weights);
    float computeWight(const float residual);
private:
    float sigma_;
    Tdistribution _tdistribution;
};

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

}

#endif // CALWEIGHT_H
