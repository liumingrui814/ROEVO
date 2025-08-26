#ifndef VISUALIZER_BASE_H
#define VISUALIZER_BASE_H

#include <pangolin/pangolin.h>

#include <thread>
#include <unordered_map>
#include <atomic>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <mutex>

class VisualizerBase
{
public:
    /** @brief 基类构造函数
     */
    VisualizerBase();
    ~VisualizerBase();

    /** 
     * @brief 绘制单独的点
     * @param v1 点数据
     * @param color 单一颜色
     * @param size 点大小
     */
    void drawPoint(cv::Point3d v1, cv::Vec3b color, int size = 1);

    /** 
     * @brief 绘制单独的线
     * @param v1 线的端点1
     * @param v2 线的端点2
     * @param color 单一颜色
     * @param size 线的宽度
     */
    void drawLine(cv::Point3d v1, cv::Point3d v2, cv::Vec3b color, int width = 1);

    /** 
     * @brief 绘制一个坐标系，x轴-红色，y轴-绿色，z轴-蓝色
     * @param Transorm 坐标系的位姿，是一个SE3
     * @param length 坐标轴长度，默认是1
     * @param width 坐标轴宽度，默认是3
     */
    void drawCoordinate(const Eigen::Matrix4d& pose, double length=1, int width=3);

    /** 
     * @brief 绘制一个纯色轨迹
     * @param poses 轨迹的位姿列表
     * @param showRotation true则是显示每个位姿点的坐标轴，false则是显示连续的轨迹线段
     * @param color 坐标轴宽度，默认是3
     */
    void drawTrajectory(const std::vector<Eigen::Matrix4d>& poses, bool showRotation,cv::Vec3b color, double axis_len = 0.1);

    /** 
     * @brief 可视化点云并做单一颜色渲染
     * @param pointcloud 点云数据
     * @param color 单一颜色
     */
    void drawPointCloudColor(const std::vector<cv::Point3d>& pointcloud, cv::Vec3b color, int pointSize = 1);

    /** 
     * @brief 可视化点云并做单一颜色渲染
     * @param pointcloud 点云数据
     * @param color 单一颜色
     */
    void drawPointCloudColorSequencial(const std::vector<cv::Point3d>& pointcloud, cv::Vec3b color, int pointSize = 1);

    /** 
     * @brief 可视化点云并沿坐标轴方向按颜色渲染
     * @param pointcloud 点云数据
     * @param color 坐标轴，可以选"x", "y", "z"三个值
     */
    void drawPointCloudAxis(const std::vector<cv::Point3d>& pointcloud, std::string axis, int pointSize = 1);

    /** 
     * @brief 可视化彩色点云并做颜色渲染
     * @param pointcloud 点云数据
     * @param color 单一颜可视化的颜色列表，需要与点云数据的列表长度一致
     */
    void drawPointCloudColorful(const std::vector<cv::Point3d>& pointcloud, 
                        const std::vector<cv::Vec3b>& colorMap, int pointSize = 1);

    
    /**
     * @brief 可视化一个相机，即一个四棱方锥，z轴指向相机平面
     * @param cameraPose 相机位姿
     * @param color 相机颜色
     * @param camSize 绘制相机的相机的大小
    */
    void drawCamera(Eigen::Matrix4d cameraPose, cv::Vec3b color, double camSize);

    /**
     * @brief 可视化一个图像，需要给定一个画布和一个cv::Mat
     * @param image 需要可视化的图像
     * @param port 给定的画布
    */
    void draw_image(const cv::Mat& image,pangolin::View* port);
    
    void start();

    void stop();

protected:
    void render_loop(){}

public:
    //多线程机制变量
    std::atomic<bool>  stop_;
    std::thread       thread;
    std::mutex        mutex_;

    std::shared_ptr<pangolin::OpenGlRenderState> s_cam_;   //内外参视角限定变量
    std::shared_ptr<pangolin::Handler3D>         handler_; //窗口交互句柄
    pangolin::View*                              d_cam_;   //交互3D视窗视图

    cv::Mat ColorTabel_JET;     //伪彩色颜色表,JET伪彩色
    cv::Mat ColorTabel_RAINBOW; //伪彩色颜色表,RAINBOW伪彩色
    cv::Mat ColorTabel_PARULA;  //伪彩色颜色表,PARULA伪彩色

private:
    /** 
     * @brief 封装私有函数, 封装了点渲染的glColor和glVertex
     */
    void addPoint(cv::Point3d v1, cv::Vec3b color);

    /** 
     * @brief 封装私有函数, 封装了线渲染的glColor和glVertex+glVertex
     */
    void addLine(cv::Point3d v1, cv::Point3d v2, cv::Vec3b color);

    /**
     * @brief 创建一个伪彩色列表，对应0~255的伪彩色，通过ColorTabel.at<cv::Vec3b>(i,0)来获得对应的伪彩色
     *        其中 |0<---i--->255|
     * @param mapType COLORMAP的类型，即0-255颜色映射的类型 
     * @return 256*1的彩色图片，按mapType赋的伪彩色
     */
    cv::Mat generateColorMap(cv::ColormapTypes mapType);
};



#endif //VISUALIZER_BASE_H