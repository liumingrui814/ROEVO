/**
 * @brief 精匹配文件，根据粗匹配的结果引入数据关联，在数据关联的基础上进行point-tangent optimization以获得更好的结果
 * @details 粗匹配得到位姿结果后，精匹配的部分直接读取位姿并作为优化的初值
 * @author 猎魔人
 * @date 2024-10-23 添加基于点线（3D-2D）残差点云配准的优化
 * @details 参考论文/repository: 
*/
#include <iostream>
#include <fstream>
#include <sophus/se3.hpp>

#include "edgeSelector.h"
#include "FrameCache.h"
#include "KeyFrame.h"
#include "utils/tum_pose.h"
#include "utils/tum_file.h"
#include "disjointSet.h"
#include "Least_squares.h"
#include "CalWeight.h"
#include "geometryFactor.h"
#include "photometricFactor.h"

#include <chrono> //-- 计时函数

#include "paramLoader.h"
// #define __DEBUG_OPTIMIZATION__

// typedef std::pair<std::vector<orderedEdgePoint>, std::vector<orderedEdgePoint>> assoResult;

//-- 本文件涉及的全局变量
float fx; 
float fy; 
float cx; 
float cy;
float kf_trans_thres;
float kf_rot_thres;
float geo_photo_ratio;
int sample_bias;


std::vector<orderedEdgePoint> associationRef2Cur(Frame& frame_ref, Frame& frame_cur, Eigen::Matrix4d T_ref_cur)
{
    //-- 为当前帧边缘创建一个bool列表，记录这些边缘是否与参考帧构成关联
    int ref_edge_num = frame_cur.mvEdges.size();

    //-- 存在 edge-wise correspondence 并且存在关联的点集
    std::vector<orderedEdgePoint> geometry_pts;

    //-- 用参考帧的有效边缘去当前帧中关当前帧的边缘，记录是否存在关联
    for(int i = 0; i < frame_cur.mvEdges.size(); ++i)
    {

        Edge& query_edge = frame_cur.mvEdges[i];
        
        //todo 这里是 inverse 还是不用 inverse
        //-- 进行edge-wise correspondence，正确关联的边缘点的 asso_edge_ID 和 asso_point_index 会存有其关联的对应点
        std::vector<int> associated_edges = frame_ref.edgeWiseCorrespondenceReproject(query_edge, T_ref_cur);
        
        if(associated_edges.empty())
            continue;

        //-- 此时 query_edge 与当前帧存在关联，确定 query_edge 参与优化的采样点
        std::vector<orderedEdgePoint> tmp_pts;
        Edge& edge = frame_cur.mvEdges[i];
        if(edge.mvSampledEdgeIndex.empty())
        {
            edge.samplingEdgeUniform(4);
        }
        const auto& index_sampled = edge.mvSampledEdgeIndex;
        //-- 获取采样的序列
        for(int j = 0; j < index_sampled.size(); ++j)
        {
            orderedEdgePoint& pt = edge.mvPoints[index_sampled[j]];
            if (pt.mbAssociated)
            {
                //-- 只存储具备关联能力的点
                tmp_pts.push_back(pt);
            }
        }
        geometry_pts.insert(geometry_pts.end(), tmp_pts.begin(), tmp_pts.end());
    }
    return geometry_pts;
}

std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> getAssociationLines2D(Frame& frame_cur, const std::vector<orderedEdgePoint> &pts1)
{
    std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> res_list(pts1.size());
    Eigen::Vector2d NaN(FLT_MAX, FLT_MAX);
    std::pair<Eigen::Vector2d, Eigen::Vector2d> resNaN;
    resNaN.first = NaN; resNaN.second = NaN;
    int ref_edge_id;
    int ref_point_index;
    for(int i = 0; i < pts1.size(); ++i){
        orderedEdgePoint pt = pts1[i];
        if(pt.mbAssociated == true){
            ref_edge_id = pt.asso_edge_ID;
            ref_point_index = pt.asso_point_index;
        }else{
            //-- 如果这个点没有几何残差，则返回NaN
            res_list[i] = resNaN;
            continue;
        }
        int edge_index;
        if(frame_cur.mmIndexMap.find(ref_edge_id) == frame_cur.mmIndexMap.end()){
            std::cout<<"\033[31m [INDEX ERROR] \033[0m"
                    <<": Encounter an edge id that doesn't exist!"<< std::endl;
            res_list[i] = resNaN;
            continue;
        }else{
            edge_index = frame_cur.mmIndexMap[ref_edge_id];
        }

        //-- 寻找一前一后两个点作为搜索到的线特征边缘点
        int pt_idx_1 = ref_point_index - 1;
        int pt_idx_2 = ref_point_index + 1;
        if(pt_idx_1 < 0){
            pt_idx_1 = 0;
        }
        if(pt_idx_2 >= frame_cur.mvEdges[edge_index].mvPoints.size()){
            pt_idx_2 = ref_point_index;
        }
        std::pair<Eigen::Vector2d, Eigen::Vector2d> res;
        orderedEdgePoint pt_1 = frame_cur.mvEdges[edge_index].mvPoints[pt_idx_1];
        orderedEdgePoint pt_2 = frame_cur.mvEdges[edge_index].mvPoints[pt_idx_2];
        res.first = Eigen::Vector2d(pt_1.x, pt_1.y);
        res.second = Eigen::Vector2d(pt_2.x, pt_2.y);
        res_list[i] = res;
    }
    return res_list;
}

Eigen::Matrix4d RegistrationCombined(const cv::Mat& img_ref,const cv::Mat& img_cur,
                           const std::vector<orderedEdgePoint> &pts_geo, 
                           const std::vector<orderedEdgePoint> &pts_pho, 
                           Frame frame_ref, Eigen::Matrix4d pose_prior)
{
    cv::Mat img_ref_gray, img_cur_gray;
    if(img_ref.channels() == 3){
        cv::cvtColor(img_ref, img_ref_gray, cv::COLOR_BGR2GRAY);
    }else{
        img_ref_gray = img_ref.clone();
    }

    if(img_cur.channels() == 3){
        cv::cvtColor(img_cur, img_cur_gray, cv::COLOR_BGR2GRAY);
    }else{
        img_cur_gray = img_cur.clone();
    }

    const int iterations = 100;
    int N_1 = pts_geo.size();
    std::vector<orderedEdgePoint> pts_total;
    pts_total.insert(pts_total.end(), pts_geo.begin(), pts_geo.end());
    //pts_total.insert(pts_total.end(), pts_pho.begin(), pts_pho.end());
    int N_2 = pts_total.size();
    // 用先验位姿初始化优化位姿
    Sophus::SE3d pose_gn = eslam_core::ConvertTrans2Sophus(pose_prior);
    double last_cost = FLT_MAX;
    double current_cost_geo, current_cost_pho;

    std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> asso_list;
    asso_list = getAssociationLines2D(frame_ref, pts_geo);

    for(int iter=0; iter<iterations; ++iter){
        // 分别定义光度误差和几何误差的雅克比矩阵
        current_cost_geo = 0;
        current_cost_pho = 0;
        Eigen::Matrix4d Trans = pose_gn.matrix();
        Eigen::Matrix3d R = Trans.block(0,0,3,3);
        Eigen::Vector3d t = Trans.block(0,3,3,1);

        //-- 统计几何残差与光度残差的增量方程以及残差，以便进行基于分布的鲁棒优化
        std::vector<eslam_core::Mat66d> H_geo_list;
        std::vector<eslam_core::Vec6d>  g_geo_list;
        std::vector<Eigen::Vector2d> r_geo_list;
        std::vector<float> weight_geo_block; //-- 每个几何残差的分布权重
        std::vector<eslam_core::Mat66d> H_pho_list;
        std::vector<eslam_core::Vec6d>  g_pho_list;
        std::vector<double> cost_pho_list;
        std::vector<float> weight_pho_block; //-- 每个光度残差的分布权重
        
        //-- 更新几何残差的雅克比矩阵
        for(int i=0; i<N_1; ++i){
        	// 传入的是边缘点，转化为Eigen的点
            Eigen::Vector3d p1_i(pts_geo[i].x_3d, pts_geo[i].y_3d, pts_geo[i].z_3d);
            //-- 找到关联的点对
            std::pair<Eigen::Vector2d, Eigen::Vector2d> asso = asso_list[i];
            //-- 如果是无效的点则略过不予优化
            if(asso.first.x()==FLT_MAX) continue;
            Eigen::Matrix<double , 6, 6> H_curr = Eigen::Matrix<double , 6, 6>::Zero();
            Eigen::Matrix<double , 6, 1> g_curr = Eigen::Matrix<double , 6, 1>::Zero();
            //double cost_curr;
            // eslam_core::calculateJacobiPointLine3D2D(asso.first, asso.second, p1_i, R, t, fx, fy, cx, cy,
            //                                        H_curr, g_curr, cost_curr);
            Eigen::Vector2d residual_curr;
            eslam_core::calculateJacobiPointLine3D2DVec(asso.first, asso.second, p1_i, R, t, fx, fy, cx, cy,
                H_curr, g_curr, residual_curr);
            double weight = pts_geo[i].score_depth;
            // 累加
            H_geo_list.push_back(H_curr * weight * weight);
            g_geo_list.push_back(g_curr * weight * weight);
            //cost_geo_list.push_back(cost_curr);
            r_geo_list.push_back(residual_curr);

            current_cost_geo += residual_curr.squaredNorm();
        }

        //-- 几何残差构建完之后，对几何残差进行卡方检验以进行加权
        // dvo_core::sigma_3_distribution dst_geo(cost_geo_list); //-- 认为cost_geo_list满足卡方分布
        // for(int i = 0; i < H_geo_list.size(); ++i){
        //     double weight_dist = dst_geo.computeWight(cost_geo_list[i]);
        //     weight_geo_block.push_back(weight_dist);
        // }
        dvo_core::robustWeight2D rw_2d(r_geo_list);
        rw_2d.computeWeights("Huber");
        weight_geo_block = rw_2d.weights;



        //-- 更新光度残差的雅克比矩阵
        for(int i=0; i<N_2; ++i){
            //-- 光度误差只需要2D的点信息以及深度
            cv::Point pt_2d(pts_total[i].x, pts_total[i].y);
            float depth = pts_total[i].depth;
            Eigen::Matrix<double , 6, 6> H_curr = Eigen::Matrix<double , 6, 6>::Zero();
            Eigen::Matrix<double , 6, 1> g_curr = Eigen::Matrix<double , 6, 1>::Zero();
            double cost_curr;
            eslam_core::accumulate_Jacbian_single_level(img_ref_gray, img_cur_gray, pose_gn, 
                                                        pt_2d, depth, fx, fy, cx, cy,
                                                        H_curr, g_curr, cost_curr);
            double weight = pts_total[i].score_depth;

            // 累加, 由于光度会有返回0（无效点）的情况，故而这里需要略过这些无效值
            if(g_curr.norm() != 0){
                H_pho_list.push_back(H_curr * weight * weight);
                g_pho_list.push_back(g_curr * weight * weight);
                cost_pho_list.push_back(cost_curr);
                current_cost_pho += cost_curr;
            }
        }

        //-- 光度残差构建完之后，对光度残差进行卡方检验以进行加权
        dvo_core::sigma_3_distribution dst_pho(cost_pho_list); //-- 认为cost_geo_list满足卡方分布
        for(int i = 0; i < H_pho_list.size(); ++i){
            double weight_dist = dst_pho.computeWight(cost_pho_list[i]);
            weight_pho_block.push_back(weight_dist);
        }
        // dvo_core::robustWeight1D rw_1d(cost_pho_list);
        // rw_1d.computeWeights("Huber");
        // weight_pho_block = rw_1d.weights;

        //-- 总的增量方程矩阵
        Eigen::Matrix<double , 6, 6> H = Eigen::Matrix<double , 6, 6>::Zero();
        Eigen::Matrix<double , 6, 1> g = Eigen::Matrix<double , 6, 1>::Zero();

        double geo_pho_ratio = geo_photo_ratio; //-- 几何残差与光度残差谁更重要(fr_2_desk 最好 0.01)
        
        //-- 将所有残差块都更新到总的增量方程矩阵中
        for(int i = 0; i < H_geo_list.size(); ++i){
            H += H_geo_list[i] * weight_geo_block[i] * geo_pho_ratio;
            g += g_geo_list[i] * weight_geo_block[i] * geo_pho_ratio;
        }
        
        for(int i = 0; i < H_pho_list.size(); ++i){
            H += H_pho_list[i] * weight_pho_block[i] / double(255 * 255 * 25);
            g += g_pho_list[i] * weight_pho_block[i] / double(255 * 255 * 25);
        }

        H /= double(H_geo_list.size() + H_pho_list.size());
        g /= double(g_geo_list.size() + g_pho_list.size());

        Eigen::Matrix<double , 6, 1> dx = H.ldlt().solve(g);// 求解dx
        if(isnan(dx[0])){
            std::cout << "result is nan"<<std::endl;
        }
        // iter>0用来控制除去第一次，因为初始current_cost、last_cost都是0
        if(iter > 0 && current_cost_geo + current_cost_pho > last_cost){
            #ifdef __DEBUG_OPTIMIZATION__
                cout<<"current cost is: "<<current_cost_geo + current_cost_pho <<" last_cost is: "<<last_cost<<endl;
            #endif
            break;
        }
        // 进行更新，这里的dx是李代数
        pose_gn = Sophus::SE3d::exp(dx) * pose_gn;
        last_cost = current_cost_geo + current_cost_pho ;
        #ifdef __DEBUG_OPTIMIZATION__
            cout << "iteration " << iter << " cost=" << std::setprecision(12) << current_cost_geo + current_cost_pho << endl;
        #endif
        if(dx.norm() < 1e-6){
            break;
        }
    }
    return pose_gn.matrix();
}

cv::Mat visualizeReprojection(const std::vector<orderedEdgePoint> &pts_geo, 
                              const std::vector<orderedEdgePoint> &pts_pho, 
                              Eigen::Matrix4d pose_ref_cur, Frame frame_ref, cv::Mat img_backGround)
{
    Eigen::Matrix3d R = pose_ref_cur.block(0,0,3,3);
    Eigen::Vector3d t = pose_ref_cur.block(0,3,3,1);
    cv::Mat img_viz = img_backGround.clone();
    //-- 对优化的当前帧点云进行重投影，重投影回参考帧
    if(img_viz.channels()==1){
        cv::cvtColor(img_viz, img_viz, cv::COLOR_GRAY2BGR);
    }

    //-- 绘制几何残差的前后投影像素
    for(int i = 0; i < pts_geo.size(); ++i){
        cv::Point geo_pixel = cv::Point(pts_geo[i].x, pts_geo[i].y);
        //-- 几何的点才有可能存在像素点关联，因而这里可视化这种像素点关联
        if(pts_geo[i].mbAssociated == true){
            int asso_edge_ID = pts_geo[i].asso_edge_ID;
            int asso_point_index = pts_geo[i].asso_point_index;
            int ref_idx = frame_ref.mmIndexMap[asso_edge_ID];
            orderedEdgePoint pt_asso = frame_ref.mvEdges[ref_idx].mvPoints[asso_point_index];
            cv::Point asso_pixel(pt_asso.x, pt_asso.y);
            cv::line(img_viz, geo_pixel, asso_pixel, cv::Scalar(160,255,255), 1, cv::LINE_AA);
            cv::circle(img_viz, asso_pixel, 1, cv::Scalar(0,255,0), -1, cv::LINE_AA);
        }
        cv::circle(img_viz, geo_pixel, 1, cv::Scalar(0,0,255), -1, cv::LINE_AA);
    }

    //-- 绘制光度误差的前后重投影像素
    for(int i = 0; i < pts_pho.size(); ++i){
        cv::Point geo_pixel = cv::Point(pts_pho[i].x, pts_pho[i].y);
        //-- 绘制原本的当前帧像素
        cv::circle(img_viz, geo_pixel, 1, cv::Scalar(0,100,255), -1, cv::LINE_AA);
        //-- 计算重投影的像素并重新绘制重投影像素
        Eigen::Vector3d pt_3d(pts_pho[i].x_3d, pts_pho[i].y_3d, pts_pho[i].z_3d);
        Eigen::Vector3d transed_point = R * pt_3d + t;
        //-- 计算重投影坐标
        double inv_z = 1.0 / transed_point[2];
        double inv_z2 = inv_z * inv_z;
        Eigen::Vector2d proj(fx * transed_point[0] / transed_point[2] + cx, 
                            fy * transed_point[1] / transed_point[2] + cy);
        cv::circle(img_viz, cv::Point(proj(0), proj(1)), 1, cv::Scalar(0,255,255), -1, cv::LINE_AA);
    }

    return img_viz;
}

cv::Mat visualizeReprojectionBias(const std::vector<orderedEdgePoint> &pts_geo, 
                              Eigen::Matrix4d pose_ref_cur, Frame frame_ref, cv::Mat img_backGround)
{
    Eigen::Matrix3d R = pose_ref_cur.block(0,0,3,3);
    Eigen::Vector3d t = pose_ref_cur.block(0,3,3,1);
    cv::Mat img_viz = img_backGround.clone();
    //-- 对优化的当前帧点云进行重投影，重投影回参考帧
    if(img_viz.channels()==1){
        cv::cvtColor(img_viz, img_viz, cv::COLOR_GRAY2BGR);
    }

    //-- 绘制几何残差的前后投影像素
    for(int i = 0; i < pts_geo.size(); ++i){
        cv::Point geo_pixel = cv::Point(pts_geo[i].x, pts_geo[i].y);
        //-- 几何的点才有可能存在像素点关联，因而这里可视化这种像素点关联
        if(pts_geo[i].mbAssociated == true){
            int asso_edge_ID = pts_geo[i].asso_edge_ID;
            int asso_point_index = pts_geo[i].asso_point_index;
            int ref_idx = frame_ref.mmIndexMap[asso_edge_ID];
            orderedEdgePoint pt_asso = frame_ref.mvEdges[ref_idx].mvPoints[asso_point_index];
            cv::Point asso_pixel(pt_asso.x, pt_asso.y);
            cv::line(img_viz, geo_pixel, asso_pixel, cv::Scalar(160,255,255), 1, cv::LINE_AA);
            cv::circle(img_viz, asso_pixel, 1, cv::Scalar(0,255,0), -1, cv::LINE_AA);
        }
        cv::circle(img_viz, geo_pixel, 1, cv::Scalar(0,255,0), -1, cv::LINE_AA);
    }

    return img_viz;
}

bool checkPoseJump(Sophus::SE3d pose)
{
    Eigen::Matrix3d R = pose.rotationMatrix();
    Eigen::Vector3d t = pose.translation();
    double t_thres = 0.10;
    double angle_thres = 15.0;
    Eigen::AngleAxisd rotationVector(R);  
    Eigen::Vector3d axis = rotationVector.axis();  
    double angle = rotationVector.angle() * 180.0 / M_PI;
    if(t_thres < t.norm() || angle > angle_thres){
        return true;
    }else{
        return false;
    }
}

int main(int argc, char **argv) {

    //-- 读取序列配置文件
    std::string tum_name = argv[1];
    param::paramHandler ph("../config/" + tum_name + ".yaml");
    
    std::string tum_dir = ph.dataset_dir;
    
    std::string path_tum = tum_dir + tum_name + "/";
    std::vector<std::string> rgb_file_seq;
    std::vector<std::string> depth_file_seq;
    std::vector<double> rgb_stamp_seq;
    std::vector<double> depth_stamp_seq;
    tum_file::getTUMsequence(path_tum,rgb_file_seq,depth_file_seq,rgb_stamp_seq,depth_stamp_seq);
    //-- 如果是ICL数据集，那rgb文件序列和depth文件序列要反一下
    // if(ph.dataset_type == "ICL"){
    //     std::vector<std::string> tmp = rgb_file_seq;
    //     rgb_file_seq = depth_file_seq;
    //     depth_file_seq = tmp;
    //     //-- 如果是ICL数据集，rgb_stamp中的时间戳要统一除以30得到真正的时间戳
    //     for(int i = 0; i < rgb_stamp_seq.size(); ++i)
    //     {
    //         rgb_stamp_seq[i] /= 30.0;
    //     }
    // }
    std::cout<<"\033[32m[dataset] \033[0m"<<": finish loading file sequence"<<std::endl;

    std::vector<double> coarse_stamps;
    std::vector<Eigen::Matrix4d> coarse_poses;
    tum_file::getGTsequence(path_tum+"poses_coarse.txt", coarse_poses, coarse_stamps);

    fx = ph.fx;
    fy = ph.fy;
    cx = ph.cx;
    cy = ph.cy;
    geo_photo_ratio = ph.fine.geo_photo_ratio;
    kf_rot_thres = ph.fine.kf_rot_thres;
    kf_trans_thres = ph.fine.kf_trans_thres;
    sample_bias = ph.fine.sample_bias;

    int N = rgb_file_seq.size();
    float mDepthMapFactor = ph.depth_scale;

    //-- 上一帧（也就是上一个关键帧）的相关信息
    Frame frame_last;
    cv::Mat rgbMat_last;
    cv::Mat grayMat_last;
    Eigen::Matrix4d pose_last_coarse;

    cv::namedWindow("edges", cv::WINDOW_NORMAL);
    std::vector<Sophus::SE3d> poseVectorCurFrame; //-- 当前帧相对于自己的参考帧的位姿变换
    std::vector<int> cur2RefFrame;                //-- 当前帧相对的参考帧在参考帧列表的index
    std::vector<Sophus::SE3d> poseVectorRefFrame; //-- 参考帧的位姿变换
    std::cout<<"start fine tracking"<<std::endl;
    Eigen::Matrix4d NaN = Eigen::MatrixXd::Zero(4,4);
    int cnt = 0;
    for(int i = 0; i < N; ++i){
        //-- 生成当前帧
        cv::Mat imgRGB = cv::imread(path_tum + rgb_file_seq[i], CV_LOAD_IMAGE_COLOR);
        cv::Mat imgDepth = cv::imread(path_tum + depth_file_seq[i], CV_LOAD_IMAGE_UNCHANGED);
        imgDepth.convertTo(imgDepth, CV_32F, mDepthMapFactor);
        
        edgeSelector selector(20.0, ph.fine.canny_low, ph.fine.canny_high);//-- 40 80 works well
        selector.processImage(imgRGB);
        Frame frame_cur(i, imgDepth, ph.fx, ph.fy, ph.cx, ph.cy, selector.mvEdges);
        frame_cur.getFineSampledPoints(sample_bias);

        //-- 得到当前帧的粗匹配位姿
        Eigen::Matrix4d pose_curr_coarse = tum_pose::getStaticGTPose(rgb_stamp_seq[i], coarse_stamps, coarse_poses);
        //-- 第一帧做参考帧
        if(cnt == 0){
            std::cout<<"assigning"<<std::endl;
            frame_last = frame_cur;
            rgbMat_last = selector.mMatRGB.clone();
            grayMat_last = selector.mMatGray.clone();
            pose_last_coarse = pose_curr_coarse;
            cnt++;
            Sophus::SE3d identity;
            poseVectorRefFrame.push_back(identity);
            poseVectorCurFrame.push_back(identity);
            cur2RefFrame.push_back(0);
            continue;
        }

        //-- 参考帧到当前帧的位姿变换
        Eigen::Matrix4d T_ref_cur_coarse = pose_last_coarse.inverse() * pose_curr_coarse;
        Sophus::SE3d pose_final;
        // std::vector<orderedEdgePoint> query_points = frame_cur.getFineSampledPoints(3);
        std::vector<orderedEdgePoint> res_asso = associationRef2Cur(frame_last, frame_cur, T_ref_cur_coarse);

        auto start_timer_asso = std::chrono::steady_clock::now();
        //-- 只有具有几何残差的点可以参与点云配准
        Eigen::Matrix4d res_trans = RegistrationCombined(rgbMat_last, imgRGB, res_asso, res_asso, frame_last, T_ref_cur_coarse);
        
        auto end_timer_asso = std::chrono::steady_clock::now();
        auto dt_asso = std::chrono::duration<double, std::milli>(end_timer_asso - start_timer_asso).count();
        std::cout<<"fine time:"<<dt_asso<<std::endl;
        pose_final = eslam_core::ConvertTrans2Sophus(res_trans);

        if(checkPoseJump(pose_final)){
            std::cout<<"\033[33m [WARNING] \033[0m"<<rgb_stamp_seq[i]<<" : pose jumped, remain with coarse result!"<<std::endl;
            pose_final = eslam_core::ConvertTrans2Sophus(T_ref_cur_coarse);
        }

        // cv::Mat img_viz = visualizeAssociationPos(frame_last, res_asso, imgRGB);
        cv::Mat img_viz = visualizeReprojectionBias(res_asso, res_trans, frame_last, imgRGB);
        cv::imshow("edges", img_viz);
        if(cv::waitKey(1)==27)break;
        // std::string img_id = rgb_file_seq[i].substr(4, rgb_file_seq[i].length() - 4);
        // cv::imwrite(path_tum + "fine/" + img_id, img_viz);
        cnt++;
        
        
        poseVectorCurFrame.push_back(pose_final);
        cur2RefFrame.push_back(poseVectorRefFrame.size()-1);

        //-- 得到当前相对于关键帧的平移与旋转
        Eigen::Matrix4d trans = pose_final.matrix();
        Eigen::Matrix3d R = trans.block(0,0,3,3);
        Eigen::Vector3d t = trans.block(0,3,3,1);
        Eigen::AngleAxisd rotation_vector(R);  
        double theta = rotation_vector.angle() * 180 / M_PI; // 转换为角度 
        double translation = t.norm();
        bool influential = false;
        //-- 4 0.03 is tested well on fr_2_desk, 
        //-- 7 0.05 is tested well on fr_3_office
        if(theta > kf_rot_thres || translation > kf_trans_thres) influential = true;
        if( influential == true){
            frame_last = frame_cur;
            pose_last_coarse = pose_curr_coarse;
            rgbMat_last = selector.mMatRGB.clone();
            grayMat_last = selector.mMatGray.clone();
            
            //pose_gn是当前关键帧关于上一个关键帧的位姿，因此当前关键帧的全局位姿可以由上一个关键帧前推
            Sophus::SE3d frame_ref_global = poseVectorRefFrame.back() * pose_final;
            poseVectorRefFrame.push_back(frame_ref_global);
        }
    }
    cv::destroyAllWindows();

    //-- 获得从单位SE3为起点的位姿列表，时间戳为rgb_stamp_seq,
    std::vector<Eigen::Matrix4d> poseGlobal;
    pcl::PointCloud<pcl::PointXYZI>::Ptr traj(new pcl::PointCloud<pcl::PointXYZI>);
    for(int i = 0; i < poseVectorCurFrame.size(); ++i){
        int refIndex = cur2RefFrame[i];
        Sophus::SE3d poseRef = poseVectorRefFrame[refIndex];
        Sophus::SE3d poseCur = poseVectorCurFrame[i];
        Sophus::SE3d poseG = poseRef * poseCur;
        Eigen::Matrix4d trans_global = poseG.matrix();
        poseGlobal.push_back(trans_global);
    }

    //-- 根据rgb_stamp_seq的时间戳寻找对应的真值位姿，并将上述的位姿列表修改为与gt共起点
    std::vector<Eigen::Matrix4d> poseList;
    std::vector<double> stamps_final;
    Eigen::Matrix4d initialBias = Eigen::MatrixXd::Identity(4,4);
    // initialBias = tum_pose::getStaticGTPose(rgb_stamp_seq[0], gt_stamps, gt_poses);//-- 第一帧位姿（原始位姿对应E）的零漂
    // std::cout<<initialBias<<std::endl;
    bool isFirstValid = false;
    for(int i = 0; i < N; ++i){
        // //-- 如果这是第一帧有效得到真值的，则以这一帧作为基准
        if(isFirstValid == false){
            isFirstValid = true;
        }
        Eigen::Matrix4d pose_curr = poseGlobal[i];
        // //-- 检查是否有不合理的位姿
        // if(pose_curr(0,3)>100 || pose_curr(1,3)>100 || pose_curr(2,3)>100){
        //     continue;
        // }
        pose_curr = initialBias * pose_curr;
        poseList.push_back(pose_curr);
        stamps_final.push_back(rgb_stamp_seq[i]);
        pcl::PointXYZI pt;
        pt.x = pose_curr(0,3);
        pt.y = pose_curr(1,3);
        pt.z = pose_curr(2,3);
        pt.intensity = poseList.size();
        traj->push_back(pt);
    }
    std::cout<<traj->points.size()<<std::endl;
    std::string storeFileName = path_tum + "poses_fine.txt";
    tum_file::saveEvaluationFiles(storeFileName, poseList, stamps_final);



    //-- 可视化一个强度点云
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (1, 1, 1, 0);
    viewer->addCoordinateSystem(0.5);
    pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> ints(traj, "intensity");
    viewer->addPointCloud<pcl::PointXYZI> (traj, ints, "undistortP");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "undistortP");
    while (!viewer->wasStopped()) {
        viewer->spinOnce();
    }

    return 0;
}
