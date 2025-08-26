#ifndef LEAST_SQUARES_H
#define LEAST_SQUARES_H

#include <Eigen/Core>
#include<Eigen/Dense>

class normalLeastSquares
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    Eigen::Matrix<double,6,6> A;
    Eigen::Matrix<double,6,1> b;

    double error;
    //-- 为了数值稳定性，需要有一个量来归一化大小，这里使用残差块的个数来维稳是比较合理的
    int vaild_constraints;

    normalLeastSquares()
    {
        A.setZero();
        b.setZero();
        error = 0.0;
        vaild_constraints = 0;
    }

    //-- 为最小二乘问题更新一个残差块
    inline void update(const Eigen::Matrix<double,6,1>& J, const float& res, const float& weight = 1.0f)
    {
        //-- factor和weight的数值关系,由于光度误差是个介于0-255的值，除以一个255是归一化，再除以一个255可能更多为了数值稳定
        float factor = weight / ( 255.0 * 255.0);

        //-- 加权之后的最小二乘问题，残差以及Hessian矩阵都要加权
        A += J * J.transpose() * factor;
        b  -= J * res * factor;

        error += res*res*factor;

        vaild_constraints += 1;
    }

    //-- 为最小二乘问题更新一个残差块
    inline void updateBlock(const Eigen::Matrix<double,6,6>& H, Eigen::Matrix<double,6,1>& g, const double& res, const double& weight = 1.0f)
    {
        //-- factor和weight的数值关系,由于光度误差是个介于0-255的值，除以一个255是归一化，再除以一个255可能更多为了数值稳定
        double factor = weight / ( 255.0 * 255.0 * 9); //-- 按块优化是9个像素一个块，一个残差

        //-- 加权之后的最小二乘问题，残差以及Hessian矩阵都要加权
        A += H * factor;
        b  += g * factor;

        error += res*res*factor;

        vaild_constraints += 1;
    }

    //-- 所有残差块都加入了之后，除以残差块的个数来维持优化的数值稳定性
    inline void finish()
    {
        A /= (double)vaild_constraints;
        b /= (double)vaild_constraints;
        error /= (double)vaild_constraints;
    }

    //-- 解最小二乘，即解增量方程
    inline void slove(Eigen::Matrix<double, 6,1>& x)
    {
        x = A.ldlt().solve(b);
    }

};
#endif // LEAST_SQUARES_H
