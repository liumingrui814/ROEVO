#include "Optimizer.h"

using namespace edge_map;

void Optimizer::optimizeAllInvolvedKFs(const edge_map::localMapPtr pLocalMap)
{
    std::vector<KeyFramePtr>& vKFs = pLocalMap->mvKeyFrames;
    std::vector<match3d_2d> matches;
    std::vector<double> weights;

    int N = vKFs.size();

    std::vector<Sophus::SE3d> poses_adjust;
    std::vector<Sophus::SE3d> poses_adjust_align;
    poses_adjust.reserve(N);
    poses_adjust_align.reserve(N);

    //-- 滑窗中的首帧关键帧做参考位姿
    int idx_mid = pLocalMap->mvKeyFrames.size()/2;
    Sophus::SE3d pose_ref_ba = pLocalMap->mvKeyFrames[idx_mid]->KF_pose_g;

    for (size_t i = 0; i < N; ++i)
    {
        int kf_id_ref = vKFs[i]->KF_ID;
        pLocalMap->getAssoFrameMergeEdge(kf_id_ref, matches, weights);
        Sophus::SE3d pose_adjust;
        optimizeSingleKFRef2Cur(vKFs.at(i), matches, weights, pose_adjust);
        //optimizeSingleKFCur2Ref(vKFs.at(i), matches, pose_ref_ba, pose_adjust);
        poses_adjust.push_back(pose_adjust);
    }

    assert(poses_adjust.size() == vKFs.size());

    //-- 重新获得这些关键帧的位姿，以窗口的第一个关键帧位姿为参考
    Sophus::SE3d pose_ref_global = vKFs[0]->KF_pose_g;
    poses_adjust_align.push_back(pose_ref_global);
    Sophus::SE3d pose_ref = poses_adjust[0];
    for (size_t i = 1; i < N; ++i)
    {
        Sophus::SE3d pose_bias = pose_ref.inverse() * poses_adjust[i];
        Sophus::SE3d pose_curr = pose_ref_global * pose_bias;
        poses_adjust_align.push_back(pose_curr);
    }

    for (size_t i = 0; i < N; ++i)
    {
        vKFs[i]->KF_pose_g = poses_adjust_align[i];
    }
}

void Optimizer::optimizeSingleKFRef2Cur(const KeyFramePtr& pKF, std::vector<match3d_2d> matches, 
                                        std::vector<double> weights, Sophus::SE3d& adjust_pose)
{
    if(matches.empty())
    {
        adjust_pose = pKF->KF_pose_g;
        return;
    }

    // * STEP-0 从关键帧里拿到内参
    float fx = pKF->mFx;
    float fy = pKF->mFy;
    float cx = pKF->mCx;
    float cy = pKF->mCy;

    Sophus::SE3d poseKF = pKF->KF_pose_g;

    // * STEP-1 根据 matches 获得残差对
    std::vector<Eigen::Vector3d> vGeometricPoints;
    std::vector<float> vScoreDepths;
    std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> vAssociatedLines;

    assert(weights.size() == matches.size());

    for (size_t n = 0; n < matches.size(); ++n)
    {
        const auto& match = matches.at(n);
        double weight = weights.at(n);

        const std::vector<cv::Point3d>& pts_3d = match.first;
        const std::vector<elementEdge> ele_edges = match.second;
        
        //-- sub step-01 获得关联的物理边缘
        std::vector<Edge> edges;
        edges.reserve(ele_edges.size());


        for (size_t i = 0; i < ele_edges.size(); ++i)
        {
            elementEdge ele_edge = ele_edges[i];
            int index = ele_edge.kf_edge_idx;
            Edge& edge = pKF->mvEdges.at(index);
            edges.push_back(edge);
        }


        //-- sub step-02 3D边缘与物理边缘关联得到点线关联
        for (size_t i = 0; i < pts_3d.size(); ++i)
        {
            //-- 寻找 pts_3d[i] 的关联边缘，必须要内部关联才算残差          
            double nearest_dist2 = 1000000.0;
            std::pair<Eigen::Vector2d, Eigen::Vector2d> associatedLine;
            float score_nearest = 0.0;

            Eigen::Vector3d pt_3d(pts_3d[i].x, pts_3d[i].y, pts_3d[i].z);
            Eigen::Vector3d pt_3d_trans = poseKF.inverse() * pt_3d;
            double x = fx * pt_3d_trans.x() / pt_3d_trans.z() + cx;
            double y = fy * pt_3d_trans.y() / pt_3d_trans.z() + cy;

            bool find = false;

            for(size_t j = 0; j < edges.size(); ++j)
            {
                Edge& edge = edges[j];
                //-- 半径内搜索
                for(int k = 0; k < edge.mvPoints.size(); ++k)
                {
                    double x_cur = edge.mvPoints[k].x;
                    double y_cur = edge.mvPoints[k].y;
                    double dist_2 = (x-x_cur)*(x-x_cur) + (y-y_cur)*(y-y_cur);
                    
                    if(dist_2 < 4 && dist_2 < nearest_dist2)
                    {
                        nearest_dist2 = dist_2;
                        // int left_idx = k-1 < 0 ? 0 : k-1;
                        int left_idx = k-1 < 0 ? 0 : k-1;
                        int right_idx = k+1 >= edge.mvPoints.size() ? edge.mvPoints.size()-1 : k+1;

                        associatedLine.first = Eigen::Vector2d(edge.mvPoints[left_idx].x, edge.mvPoints[left_idx].y);
                        associatedLine.second = Eigen::Vector2d(edge.mvPoints[right_idx].x, edge.mvPoints[right_idx].y);
                        score_nearest = edge.mvPoints[k].score_depth * weight;
                        //score_nearest = edge.mvPoints[k].score_depth;
                        
                        find = true;

                    }
                }
            }
            
            //-- 获得最近的点对的信息
            if(find)
            {
                vGeometricPoints.push_back(pt_3d);
                vAssociatedLines.push_back(associatedLine);
                vScoreDepths.push_back(score_nearest);
            }
        }
    }

    // * STEP-2 根据残差对进行点-线 雅可比的优化
    Sophus::SE3d poseKF_inv = poseKF.inverse();
    RegistrationGeometricParallel(vGeometricPoints, vScoreDepths, vAssociatedLines, poseKF_inv, fx, fy, cx, cy);

    // * STEP-3 更新当前的位姿
    adjust_pose = poseKF_inv.inverse();
}

void Optimizer::optimizeSingleKFCur2Ref(const KeyFramePtr& pKF, std::vector<match3d_2d> matches,
                                        const Sophus::SE3d& pose_ref, Sophus::SE3d& adjust_pose)
{
    if(matches.empty())
    {
        adjust_pose = pKF->KF_pose_g;
        return;
    }

    // * STEP-0 从关键帧里拿到内参
    float fx = pKF->mFx;
    float fy = pKF->mFy;
    float cx = pKF->mCx;
    float cy = pKF->mCy;

    Sophus::SE3d poseKF = pKF->KF_pose_g;
    Sophus::SE3d pose_ref_cur = pose_ref.inverse() * poseKF;

    // * STEP-1 根据 matches 获得残差对
    std::vector<Eigen::Vector3d> vGeometricPoints;
    std::vector<float> vScoreDepths;
    std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> vAssociatedLines;
    
    cv::Mat img_viz_gray = pKF->mMatGray.clone();
    cv::Mat img_viz;
    // cv::cvtColor(img_viz_gray, img_viz, cv::COLOR_GRAY2BGR);
    // cv::namedWindow("dbg", cv::WINDOW_NORMAL);

    std::vector<cv::Point2d> pts_2d_ref_total;

    for (const auto& match : matches)
    {
        const std::vector<cv::Point3d>& pts_3d = match.first;
        const std::vector<elementEdge> ele_edges = match.second;

        //-- sub step-00 投影得到3D局部地图的2D边缘
        std::vector<cv::Point2d> pts_2d_ref;
        pts_2d_ref.reserve(pts_3d.size());

        for(size_t i = 0; i < pts_3d.size(); ++i)
        {
            Eigen::Vector3d pt_3d(pts_3d[i].x, pts_3d[i].y, pts_3d[i].z);
            Eigen::Vector3d pt_3d_trans = pose_ref.inverse() * pt_3d;
            double x = fx * pt_3d_trans.x() / pt_3d_trans.z() + cx;
            double y = fy * pt_3d_trans.y() / pt_3d_trans.z() + cy;
            pts_2d_ref.emplace_back(x,y);
        }
        pts_2d_ref_total.insert(pts_2d_ref_total.end(), pts_2d_ref.begin(), pts_2d_ref.end());
        
        //-- sub step-01 获得关联的物理边缘 (3D边缘)
        std::vector<cv::Point3d> pts_3d_curr;
        std::vector<double> pts_score_curr;

        for (size_t i = 0; i < ele_edges.size(); ++i)
        {
            elementEdge ele_edge = ele_edges[i];
            int index = ele_edge.kf_edge_idx;
            Edge& edge = pKF->mvEdges.at(index);

            if(edge.mvSampledEdgeIndex.empty())
            {
                std::cout<<"sample uniform"<<std::endl;
                edge.samplingEdgeUniform(3);
            }

            const std::vector<int>& sample_indices = edge.mvSampledEdgeIndex;
            for(size_t j = 0; j < sample_indices.size(); ++j)
            {
                int index = sample_indices[j];
                const orderedEdgePoint& pt = edge.mvPoints[index];
                cv::Point3d pt_3d_curr(pt.x_3d, pt.y_3d, pt.z_3d);
                pts_3d_curr.emplace_back(pt_3d_curr);
                pts_score_curr.push_back(pt.score_depth);
            }
        }


        //-- sub step-02 3D边缘与物理边缘关联得到点线关联
        for (size_t i = 0; i < pts_3d_curr.size(); ++i)
        {
            //-- 寻找 pts_3d_curr[i] 的关联边缘，必须要内部关联才算残差          
            double nearest_dist2 = 1000000.0;
            std::pair<Eigen::Vector2d, Eigen::Vector2d> associatedLine;

            //-- 计算当前帧3D边缘点投影到参考帧上的坐标(x,y)
            Eigen::Vector3d pt_3d(pts_3d_curr[i].x, pts_3d_curr[i].y, pts_3d_curr[i].z);
            Eigen::Vector3d pt_3d_trans = pose_ref_cur * pt_3d;
            double x = fx * pt_3d_trans.x() / pt_3d_trans.z() + cx;
            double y = fy * pt_3d_trans.y() / pt_3d_trans.z() + cy;

            cv::Point pt(x,y);
            cv::circle(img_viz, pt, 1, cv::Scalar(0,255,0), 2, cv::LINE_AA);

            bool find = false;

            for(int j = 0; j < pts_2d_ref.size(); ++j)
            {
                double x_cur = pts_2d_ref[j].x;
                double y_cur = pts_2d_ref[j].y;
                double dist_2 = (x-x_cur)*(x-x_cur) + (y-y_cur)*(y-y_cur);
                
                if(dist_2 < 4 && dist_2 < nearest_dist2)
                {
                    nearest_dist2 = dist_2;
                    // int left_idx = k-1 < 0 ? 0 : k-1;
                    int left_idx = j-1 < 0 ? 0 : j-1;
                    int right_idx = j+1 >= pts_2d_ref.size() ? pts_2d_ref.size()-1 : j+1;

                    associatedLine.first = Eigen::Vector2d(pts_2d_ref[left_idx].x, pts_2d_ref[left_idx].y);
                    associatedLine.second = Eigen::Vector2d(pts_2d_ref[right_idx].x, pts_2d_ref[right_idx].y);
                    
                    find = true;
                }
            }
            
            //-- 获得最近的点对的信息
            if(find)
            {
                vGeometricPoints.push_back(pt_3d);
                vAssociatedLines.push_back(associatedLine);
                vScoreDepths.push_back(pts_score_curr[i]);
            }
        }
    }

    // * STEP-2 根据残差对进行点-线 雅可比的优化
    RegistrationGeometricParallel(vGeometricPoints, vScoreDepths, vAssociatedLines, pose_ref_cur, fx, fy, cx, cy);

    // * STEP-3 更新当前的位姿
    adjust_pose = pose_ref * pose_ref_cur;
}

void Optimizer::RegistrationGeometricParallel(std::vector<Eigen::Vector3d> vGeometricPoints,
            std::vector<float> vScoreDepths,
            std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> vAssociatedLines,
            Sophus::SE3d& pose_cur_ref, float fx, float fy, float cx, float cy)
{

    const int iterations = 100;
    int N_1 = vGeometricPoints.size();

    double last_cost = FLT_MAX;
    double current_cost_geo;

    for(int iter=0; iter<iterations; ++iter)
    {
        //-- 几何误差的雅克比矩阵
        current_cost_geo = 0;

        //-- 统计几何残差与光度残差的增量方程以及残差，以便进行基于分布的鲁棒优化
        std::vector<fine::Mat66d> H_geo_list;
        std::vector<fine::Vec6d>  g_geo_list;
        std::vector<Eigen::Vector2d> residual_geo_list;
        std::vector<float> weight_geo_list; //-- 每个几何残差的分布权重
        
        // //-- 更新几何残差的雅克比矩阵
        current_cost_geo = tbb::parallel_reduce(
            tbb::blocked_range<size_t>(0, N_1),  // 迭代范围
            0.0,                                 // 初始值
            [&](const tbb::blocked_range<size_t>& r, double local_cost) {
                // 本地存储每个线程的计算结果
                std::vector<Eigen::Matrix<double, 6, 6>> local_H;
                std::vector<Eigen::Matrix<double, 6, 1>> local_g;
                std::vector<Eigen::Vector2d> local_residual;
                
                for(size_t i = r.begin(); i != r.end(); ++i) {
                    // 传入的是边缘点，转化为Eigen的点
                    Eigen::Vector3d p1_i = vGeometricPoints[i];
                    
                    // 跳过无效点
                    const auto& asso = vAssociatedLines[i];
                    if(asso.first.x() == FLT_MAX) continue;
                    
                    // 计算雅可比矩阵
                    Eigen::Matrix<double, 6, 6> H_curr = Eigen::Matrix<double, 6, 6>::Zero();
                    Eigen::Matrix<double, 6, 1> g_curr = Eigen::Matrix<double, 6, 1>::Zero();
                    Eigen::Vector2d r_curr;
                    calculateJacobiPointLine3D2D(asso.first, asso.second, p1_i, H_curr, g_curr, r_curr, pose_cur_ref, fx, fy, cx, cy);
                    
                    // 加权
                    double weight = vScoreDepths[i];
                    double weight_sq = weight * weight;
                    
                    // 累加到本地存储
                    local_H.push_back(H_curr * weight_sq);
                    local_g.push_back(g_curr * weight_sq);
                    local_residual.push_back(r_curr);
                    
                    // 累加本地cost
                    local_cost += r_curr.squaredNorm();
                }
                
                // 将本地结果合并到全局（加锁或使用线程安全操作）
                {
                    static std::mutex mtx;
                    std::lock_guard<std::mutex> lock(mtx);
                    
                    H_geo_list.insert(H_geo_list.end(), local_H.begin(), local_H.end());
                    g_geo_list.insert(g_geo_list.end(), local_g.begin(), local_g.end());
                    residual_geo_list.insert(residual_geo_list.end(), local_residual.begin(), local_residual.end());
                }
                
                return local_cost;
            },
            // 合并各个线程的local_cost
            [](double a, double b) { return a + b; }
        );

        //-- 几何残差构建完之后，对几何残差进行卡方检验以进行加权
        eslam_core::robustWeight2D dst_geo(residual_geo_list);
        dst_geo.computeWeights("Huber");
        weight_geo_list = dst_geo.weights;

        //-- 总的增量方程矩阵
        Eigen::Matrix<double , 6, 6> H = Eigen::Matrix<double , 6, 6>::Zero();
        Eigen::Matrix<double , 6, 1> g = Eigen::Matrix<double , 6, 1>::Zero();
        
        //-- 将所有残差块都更新到总的增量方程矩阵中
        for(int i = 0; i < H_geo_list.size(); ++i){
            H += H_geo_list[i] * weight_geo_list[i];
            g += g_geo_list[i] * weight_geo_list[i];
        }

        H /= double(H_geo_list.size());
        g /= double(g_geo_list.size());

        Eigen::Matrix<double , 6, 1> dx = H.ldlt().solve(g);// 求解dx
        if(isnan(dx[0])){
            std::cout << "result is nan"<<std::endl;
        }
        // iter>0用来控制除去第一次，因为初始current_cost、last_cost都是0
        if(iter > 0 && current_cost_geo > last_cost){
            break;
        }
        // 进行更新，这里的dx是李代数
        pose_cur_ref = Sophus::SE3d::exp(dx) * pose_cur_ref;
        // last_cost = current_cost_geo + current_cost_pho ;
        last_cost = current_cost_geo;

        if(dx.norm() < 1e-6){
            break;
        }
    }
}

std::pair<Eigen::Vector2d, Eigen::Vector2d> 
Optimizer::normalizeLinePoints2D(Eigen::Vector2d startPointRaw, Eigen::Vector2d endPointRaw)
{
    Eigen::Vector2d dirRaw = endPointRaw - startPointRaw;
    Eigen::Vector2d dirNorm = dirRaw/dirRaw.norm();
    Eigen::Vector2d startPoint = startPointRaw;
    Eigen::Vector2d endPoint = startPoint + dirNorm;
    std::pair<Eigen::Vector2d, Eigen::Vector2d> linePair;
    linePair.first = startPoint; linePair.second = endPoint;
    return linePair;
}

void Optimizer::calculateJacobiPointLine3D2D(Eigen::Vector2d startPoint, 
                                             Eigen::Vector2d endPoint, 
                                             Eigen::Vector3d queryPoint_3d,
                                             fine::Mat66d& H_out, fine::Vec6d& g_out, Eigen::Vector2d& residual,
                                             Sophus::SE3d& pose, float fx, float fy, float cx, float cy)
{
    //-- normalize border points.
    std::pair<Eigen::Vector2d, Eigen::Vector2d> normPoints = normalizeLinePoints2D(startPoint,endPoint);
    
    //-- defination of registrated line.
    Eigen::Vector2d a = normPoints.first;
    Eigen::Vector2d b = normPoints.second;
    Eigen::Vector2d l_ba = a-b;

    fine::Mat66d H = Eigen::Matrix<double , 6, 6>::Zero();
    fine::Vec6d  g = Eigen::Matrix<double , 6, 1>::Zero();

    //-- 计算3D空间的位姿变换
    
    Eigen::Vector3d transed_point = pose * queryPoint_3d;
    //-- 计算重投影坐标
    double inv_z = 1.0 / transed_point[2];
    double inv_z2 = inv_z * inv_z;
    Eigen::Vector2d proj(fx * transed_point[0] / transed_point[2] + cx, 
                         fy * transed_point[1] / transed_point[2] + cy);

    
    //-- 3D位姿变换产生的3D-2D投影的位姿残差的雅克比矩阵为2x6
    Eigen::Matrix<double , 2, 6> J_orig = Eigen::Matrix<double , 2, 6>::Zero();
    Eigen::Matrix<double , 2, 6> J = Eigen::Matrix<double , 2, 6>::Zero();
    //-- 一般重投影误差的2x6残差矩阵, fx/z如果是负的说明这是 关联像素-重投影像素，正的则是 重投影像素-关联像素
    J_orig << -fx * inv_z,
        0,
        fx * transed_point[0] * inv_z2,
        fx * transed_point[0] * transed_point[1] * inv_z2,
        -fx - fx * transed_point[0] * transed_point[0] * inv_z2,
        fx * transed_point[1] * inv_z,
        0,
        -fy * inv_z,
        fy * transed_point[1] * inv_z,
        fy + fy * transed_point[1] * transed_point[1] * inv_z2,
        -fy * transed_point[0] * transed_point[1] * inv_z2,
        -fy * transed_point[0] * inv_z;
    //-- 由于根据下面的计算是l_dc是残差故而J_orig要反一下
    J_orig = -J_orig;
    Eigen::Matrix<double , 2, 3> J_orig_t = J_orig.block(0,0,2,3);
    Eigen::Matrix<double , 2, 3> J_orig_R = J_orig.block(0,3,2,3);
    
    //-- 图示计算残差的三角形，残差向量为l_dc，c为位姿变换后的query point
    //    @a----------@d---@b
    //      --        |   /
    //         --     |  /
    //            --  | /
    //                @c
    //-- dr/dt = d(l_dc)/dt = d(l_bc)/dt - d(l_bd)/dt
    //--  l_bd = l_ba * (l_ba^T . l_bc), l_bc = pc - pb = warp(q) - p_b
    //-- dr/dt = (I - l_ba * l_ba^T) J_3d2d

    //-- 点线残差构造时的此项都是固定的一个2x2的倍数矩阵：(I - l_ba * l_ba^T)
    Eigen::Matrix<double, 2, 2> k_line = Eigen::Matrix<double , 2, 2>::Zero();
    k_line = Eigen::Matrix2d::Identity() - l_ba * l_ba.transpose();

    //-- 平移部分的量
    J.block(0,0,2,3) = k_line * J_orig_t;
    
    //-- 对旋转的雅克比矩阵, 形式上与平移相同，只是多一个Rq+t对旋转的求导
    //--      dr/dR = d(l_dc)/dR = d(l_bc)/dR - d(l_bd)/dR
    //-- d(l_bc)/dR = d warp(q)/dR
    //-- d(l_bd)/dR = d[l_ba * (l_ba^T . l_bc)]/dR = -l_ba * l_ba^T * Rq^
    J.block(0,3,2,3) = k_line * J_orig_R;
    
    //-- 计算残差 r = l_bc - l_bd
    Eigen::Vector2d l_bc = proj - b;
    Eigen::Vector2d l_bd = l_ba * (l_ba.transpose() * l_bc);
    Eigen::Vector2d r = l_bc - l_bd;

    //-- update derivative matrix
    residual = r;
    g = -J.transpose() * r; g_out = g;
    H = J.transpose() * J;  H_out = H;
}