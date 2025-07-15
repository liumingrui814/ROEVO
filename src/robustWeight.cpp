#include "robustWeight.h"

using namespace eslam_core;

//-- 构造函数
robustWeight2D::robustWeight2D(const std::vector<Eigen::Vector2d>& input_vectors)
{
    residuals = input_vectors;
    computeStatistics();
    computeMahalanobis();
}

//-- 计算期望和标准差
void robustWeight2D::computeStatistics() 
{
    if (residuals.empty()) 
    {
        throw std::runtime_error("Input vector is empty");
    }

    // 计算期望（均值）
    expt = Eigen::Vector2d::Zero();
    for (const auto& vec : residuals) {
        expt += vec;
    }
    expt /= static_cast<float>(residuals.size());

    // 计算标准差
    sigma = Eigen::Vector2d::Zero();
    for (const auto& vec : residuals) {
        Eigen::Vector2d diff = vec - expt;
        sigma += diff.cwiseProduct(diff);
    }
    sigma = (sigma / static_cast<float>(residuals.size())).cwiseSqrt();
}

void robustWeight2D::computeMahalanobis()
{
    // 预分配结果向量以避免动态扩容
    Mahalanobis.reserve(residuals.size());
    
    // 计算逆协方差矩阵 (假设两个维度不相关)
    // 马氏距离公式: sqrt(r' * Sigma^-1 * r)
    // 对于对角协方差矩阵，可以简化为 sqrt((r1/s1)^2 + (r2/s2)^2)
    const double inv_sigma1_sq = 1.0 / (sigma.x() * sigma.x());
    const double inv_sigma2_sq = 1.0 / (sigma.y() * sigma.y());
    
    // 遍历所有残差
    for (const auto& r : residuals) 
    {
        // 计算马氏距离平方
        const double dist_sq = r[0] * r[0] * inv_sigma1_sq 
                             + r[1] * r[1] * inv_sigma2_sq;
        // 存储平方根结果
        Mahalanobis.emplace_back(static_cast<float>(std::sqrt(dist_sq)));
    }
}

void robustWeight2D::computeWeights(const std::string& type)
{
    // 清空之前的权重
    weights.clear();
    weights.reserve(Mahalanobis.size());

    // 检查类型是否有效
    assert(type == "Cauchy" || type == "Huber" || type == "Tukey");

    // 根据类型选择权重函数
    for (float d : Mahalanobis) {
        if (type == "Cauchy") {
            weights.push_back(rwUtils::CauchyWeight(d));
        } else if (type == "Huber") {
            weights.push_back(rwUtils::HuberWeight(d));
        } else if (type == "Tukey") {
            weights.push_back(rwUtils::TukeyWeight(d));
        }
    }
}


robustWeight1D::robustWeight1D(const std::vector<float>& input_vectors)
{
    residuals = input_vectors;
    computeStatistics();
    computeMahalanobis();
}

void robustWeight1D::computeStatistics()
{
    if (residuals.empty()) 
    {
        throw std::runtime_error("Input vector is empty");
    }

    // 计算期望（均值）
    expt = 0.0f;
    for (const auto& vec : residuals) {
        expt += vec;
    }
    expt /= static_cast<float>(residuals.size());

    // 计算标准差
    sigma = 0.0f;
    for (const auto& vec : residuals) {
        float diff = vec - expt;
        sigma += diff * diff;
    }
    sigma = (sigma / static_cast<float>(residuals.size()));
}

void robustWeight1D::computeMahalanobis()
{
    // 预分配结果向量以避免动态扩容
    Mahalanobis.reserve(residuals.size());
    
    // 一维残差的马氏距离就是 x / sigma
    const double inv_sigma = 1.0 / sigma;
    
    // 遍历所有残差
    for (const auto& r : residuals) 
    {
        // 存储马氏距离
        Mahalanobis.emplace_back(r * inv_sigma);
    }
}

void robustWeight1D::computeWeights(const std::string& type)
{
    // 清空之前的权重
    weights.clear();
    weights.reserve(Mahalanobis.size());

    // 检查类型是否有效
    assert(type == "Cauchy" || type == "Huber" || type == "Tukey");

    // 根据类型选择权重函数
    for (float d : Mahalanobis) {
        if (type == "Cauchy") {
            weights.push_back(rwUtils::CauchyWeight(d));
        } else if (type == "Huber") {
            weights.push_back(rwUtils::HuberWeight(d));
        } else if (type == "Tukey") {
            weights.push_back(rwUtils::TukeyWeight(d));
        }
    }
}

void robustWeightChi2::computeStatistics()
{
    if (residuals_2.empty()) 
    {
        sigma2 = 0.0;
        chi2residuals_2.clear();
        return;
    }

    // 1. 计算残差平方的均值 (sigma^2 = E[epsilon^2])
    double sum;
    for (const auto& r2 : residuals_2)
    {
        sum += r2;
    }
    sigma2 = sum/static_cast<double>(residuals_2.size());

    // 2. 标准化残差: residuals_2[i] / sigma_2 -> 近似服从 chi2(1)
    chi2residuals_2.resize(residuals_2.size());
    for (size_t i = 0; i < residuals_2.size(); ++i) 
    {
        chi2residuals_2[i] = residuals_2[i] / sigma2;
    }
}

void robustWeightChi2::computeWights()
{
    if (chi2residuals_2.empty())
    {
        return;
    }
    weight.reserve(chi2residuals_2.size());
    for (const auto& chi2_r2 : chi2residuals_2)
    {
        if (chi2_r2 < CHI2_20_PERCENT){
            weight.push_back(1.0);
        } 
        else if (chi2_r2 < CHI2_10_PERCENT){
            weight.push_back(0.8);
        }
        else if (chi2_r2 < CHI2_05_PERCENT){
            weight.push_back(0.2);
        }
        else if (chi2_r2 < CHI2_01_PERCENT){
            weight.push_back(0.05);
        }
        else{
            weight.push_back(0);
        }
    }
}

double robustWeightChi2::computeWight(double error)
{
    error /= sigma2;
    if (error < CHI2_20_PERCENT){
        return 1.0;
    } 
    else if (error < CHI2_10_PERCENT){
        return 0.8;
    }
    else if (error < CHI2_05_PERCENT){
        return 0.2;
    }
    else if (error < CHI2_01_PERCENT){
        return 0.05;
    }
    else{
        return 0.0;
    }
}

void sigma_3_distribution::constructDistribution()
{
    //-- 计算期望
    double sum = 0.0;
    for (double value : residuals){
        sum += value;
    }
    expt = sum / residuals.size();
    //-- 计算标准差
    double sumOfSquares = 0.0;
    for (double value : residuals){
        sumOfSquares += (value - expt) * (value - expt);
    }
    sigma = std::sqrt(sumOfSquares / residuals.size());
    //-- 计算残差绝对值的中位数
    std::vector<double> absoluteValues;
    for (double value : residuals){
        absoluteValues.push_back(std::abs(value));
    }
    std::sort(absoluteValues.begin(), absoluteValues.end());
    if (absoluteValues.size() % 2 == 0) {
        abs_med = (absoluteValues[absoluteValues.size() / 2 - 1] + absoluteValues[absoluteValues.size() / 2]) / 2.0;
    }else{
        abs_med = absoluteValues[absoluteValues.size() / 2];
    }
}

double sigma_3_distribution::computeWight(double error)
{
    double loss = std::abs(error - expt);
    if(loss < sigma){
        return 1;
    }else if(loss < 2 * sigma){
        return 0.5;
    }else{
        return 0.2;
    }
}

