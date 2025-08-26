#include "featureMerger.h"

using namespace edge_map;

double featureMerger::statisticFilter(std::vector<Eigen::Vector3d>& point_cluster, std::vector<float>& scores, double std_mult) 
{
    if (point_cluster.empty() || point_cluster.size()<4) return 1.0;

    // 1. 计算每个点到所有其他点的平均距离
    std::vector<double> avg_distances;
    avg_distances.reserve(point_cluster.size());

    for (const auto& point : point_cluster) {
        double total_distance = 0.0;
        
        for (const auto& other_point : point_cluster) {
            if (&point != &other_point) { // 不与自己比较
                total_distance += (point - other_point).norm();
            }
        }
        
        double avg_distance = total_distance / (point_cluster.size() - 1);
        avg_distances.push_back(avg_distance);
    }

    // 2. 计算距离的均值 (μ)
    double mean = std::accumulate(avg_distances.begin(), avg_distances.end(), 0.0) / avg_distances.size();

    // 3. 计算方差 (σ²)
    double variance = 0.0;
    for (double dist : avg_distances) {
        variance += (dist - mean) * (dist - mean);
    }
    variance /= avg_distances.size();

    // 4. 计算标准差 (σ)
    double std_dev = std::sqrt(variance);
    // std::cout<<std_dev<<std::endl;
    // if (std_dev > 0.004)
    // {
    //     //-- 标准差太大，直接清空不予拟合
    //     point_cluster.clear();
    //     return;
    // }

    // 5. 设置过滤阈值
    double threshold = mean + std_mult * std_dev;

    // 6. 过滤离群点
    std::vector<Eigen::Vector3d> filtered_points;
    std::vector<float> scores_culled;
    filtered_points.reserve(point_cluster.size());
    scores_culled.reserve(point_cluster.size());

    for (size_t i = 0; i < point_cluster.size(); ++i) {
        if (avg_distances[i] < threshold) {
            filtered_points.push_back(point_cluster[i]);
            scores_culled.push_back(scores[i]);
        }
    }

    // 7. 替换原始点云
    point_cluster = std::move(filtered_points);
    scores = std::move(scores_culled);
    return threshold;
}

void featureMerger::getMergedClusterProjection(const std::vector<elementEdge>& vElements, 
    const std::vector<int>& vKFindices,
    const std::vector<KeyFramePtr>& vKeyFrames,
    std::vector<cv::Point3d>& mergedCloud_global,
    std::vector<int>& involved_elements)
{
    assert(vElements.size()==vKFindices.size());
    assert(vKeyFrames.size()>0);

    mergedCloud_global.clear();
    
    //-- 从关键帧里提取出内参
    float fx = vKeyFrames[0]->mFx;
    float fy = vKeyFrames[0]->mFy;
    float cx = vKeyFrames[0]->mCx;
    float cy = vKeyFrames[0]->mCy;

    // * STEP-1 保存这个 cluster 的边缘以及每个边缘的位姿，并顺便找到最长的边缘
    std::vector<Edge> vEdges;
    std::vector<Sophus::SE3d> vKFposes;
    int N = vElements.size();
    vEdges.reserve(N);
    vKFposes.reserve(N);

    int longest_idx = -1;
    int longest_size = 0;
    for(size_t i = 0; i < N; ++i)
    {
        int ele_edge_idx = vElements[i].kf_edge_idx;
        Edge& edge = vKeyFrames[vKFindices[i]]->mvEdges[ele_edge_idx];
        vEdges.push_back(edge);
        if(edge.mvPoints.size() > longest_size)
        {
            longest_idx = vEdges.size() - 1;
            longest_size = edge.mvPoints.size();
        }

        Sophus::SE3d pose = vKeyFrames[vKFindices[i]]->KF_pose_g;
        vKFposes.push_back(pose);
    }

    // * STEP-2 对最长的边缘进行采样
    std::vector<Eigen::Vector3d> sampled_longest_edge;
    const int sampleBias = 3;
    Edge& longestEdge = vEdges[longest_idx]; 
    Sophus::SE3d pose_ref = vKFposes[longest_idx];
    auto ptToEigen = [](const orderedEdgePoint& pt) {
        return Eigen::Vector3d(pt.x_3d, pt.y_3d, pt.z_3d);
    };

    sampled_longest_edge.reserve(longest_size / sampleBias + 2);

    sampled_longest_edge.push_back(ptToEigen(longestEdge.mvPoints.front()));  // 第一个点
    for(size_t i = sampleBias; i < longest_size - 1; i += sampleBias) 
    {
        sampled_longest_edge.push_back(ptToEigen(longestEdge.mvPoints[i]));
    }
    if(longest_size > 1) {  // 最后一个点
        sampled_longest_edge.push_back(ptToEigen(longestEdge.mvPoints.back()));
    }

    //-- 将采样的参考边缘变换到全局坐标系下
    for (size_t i = 0; i < sampled_longest_edge.size(); ++i)
    {
        sampled_longest_edge[i] = pose_ref * sampled_longest_edge[i];
    }

    // * STEP-3 最长的边缘通过投影搜索其他边缘与之关联的边缘点
    size_t N_edge = sampled_longest_edge.size();
    std::vector<std::vector<Eigen::Vector3d>> point_neighbors_total;
    std::vector<std::vector<float>> point_neighbors_score_total;
    point_neighbors_total.resize(N_edge);
    point_neighbors_score_total.resize(N_edge);

    //-- 统计每个 elementEdge 构成 merged cloud 的比例
    std::vector<int> statistics(N, 0);
    
    for(size_t i = 0; i < N_edge; ++i)
    {
        //-- 参考边缘的第 i 个点
        Eigen::Vector3d pt = sampled_longest_edge[i];
        //-- 参考边缘第 i 个点对应的相邻点
        std::vector<Eigen::Vector3d> point_neighbors;
        //-- 参考边缘第 i 个点对应的相邻点的分数
        std::vector<float> point_scores;
        
        //-- 将第 i 个点分别投影到 cluster 中各边缘所在的帧上，查找关联点
        for(size_t j = 0; j < N; ++j)
        {
            if (j == longest_idx) continue; //-- 自身不处理
            
            Edge& edge = vEdges[j];
            Sophus::SE3d pose_cur = vKFposes[j];
            Sophus::SE3d pose_cur_inv = pose_cur.inverse();
            
            //-- 参考边缘点重投影到当前帧
            Eigen::Vector3d pt_trans = pose_cur_inv * pt;
            double x = fx * pt_trans.x() / pt_trans.z() + cx;
            double y = fy * pt_trans.y() / pt_trans.z() + cy;

            //-- 半径内搜索
            std::vector<Eigen::Vector3d> neighbors;
            std::vector<float> scores;
            for(size_t k = 0; k < edge.mvPoints.size(); ++k)
            {
                double x_cur = edge.mvPoints[k].x;
                double y_cur = edge.mvPoints[k].y;
                double dist_2 = (x-x_cur)*(x-x_cur) + (y-y_cur)*(y-y_cur);
                if(dist_2 < 4)
                {
                    Eigen::Vector3d pt_3d(edge.mvPoints[k].x_3d, edge.mvPoints[k].y_3d, edge.mvPoints[k].z_3d);
                    pt_3d = pose_cur * pt_3d; //-- 投影到世界坐标系下
                    neighbors.push_back(pt_3d);
                    scores.push_back(edge.mvPoints[k].score_depth);
                }
            }
            point_neighbors.insert(point_neighbors.end(), neighbors.begin(), neighbors.end());
            point_scores.insert(point_scores.end(), scores.begin(), scores.end());

            //-- 更新统计
            statistics[j] += neighbors.size();
        }
        point_neighbors_total[i] = point_neighbors;
        point_neighbors_score_total[i] = point_scores;
    }

    // * STEP-4 计算质心（全局坐标系下）
    std::vector<cv::Point3d> centroid_global;
    for(size_t i = 0; i < N_edge; ++i)
    {
        std::vector<Eigen::Vector3d> neighbors = point_neighbors_total[i];
        std::vector<float> scores = point_neighbors_score_total[i];
        Eigen::Vector3d current = sampled_longest_edge[i];
        
        std::vector<Eigen::Vector3d> cluster = neighbors;
        cluster.push_back(current);
        scores.push_back(1);

        Eigen::Vector3d centroid(0.0, 0.0, 0.0);

        float sum_total = 0.0f;

        for (size_t j = 0; j < cluster.size(); ++j)
        {
            centroid.x() += cluster[j].x() * scores[j];
            centroid.y() += cluster[j].y() * scores[j];
            centroid.z() += cluster[j].z() * scores[j];
            sum_total += scores[j];
        }
        
        int numPoints = cluster.size();
        if(numPoints <= 1)
        {
            //centroid_global.push_back(NaN);
            continue;
        }
        centroid.x() /= sum_total;
        centroid.y() /= sum_total;
        centroid.z() /= sum_total;
        cv::Point3d p3d_cv(centroid.x(), centroid.y(), centroid.z());
        centroid_global.push_back(p3d_cv);
    }
    if(!centroid_global.empty())
    {
        mergedCloud_global = std::move(centroid_global);
    }

    // * STEP-5 整理构成关系
    involved_elements.clear();
    involved_elements.reserve(N);
    //-- 参考边缘肯定构成
    involved_elements.push_back(longest_idx);
    for (size_t i = 0; i < statistics.size(); ++i)
    {
        if(statistics[i] > 5 || statistics[i] >= 0.5 * vEdges[i].mvPoints.size())
        {
            involved_elements.push_back(i);
        }
    }
}

std::tuple<cv::Point2d, bool> featureMerger::calculateFootAndCheck(cv::Point2d A, cv::Point2d B, cv::Point2d C) 
{
    
    //-- 计算向量AB和AC
    cv::Point2d AB = B - A;
    cv::Point2d AC = C - A;
    
    //-- 计算AB长度的平方
    double AB_length_squared = AB.x * AB.x + AB.y * AB.y;
    //-- 计算投影比例 (AC·AB)/(AB·AB)
    double projection = (AC.x * AB.x + AC.y * AB.y) / AB_length_squared;
    //-- 计算垂足坐标
    cv::Point2d foot = A + AB * projection;
    //-- 判断垂足是否在线段AB上
    bool is_between = (projection >= 0.0) && (projection <= 1.0);
    
    return std::make_tuple(foot, is_between);
}

std::vector<size_t> featureMerger::findLongestFalseSegment(const std::vector<bool>& asso_list)
{

    if (asso_list.empty()) return {};

    size_t max_start = 0;
    size_t max_length = 0;
    size_t current_start = 0;
    bool in_segment = false;

    for (size_t i = 0; i < asso_list.size(); ++i) {
        if (!asso_list[i]) 
        {
            if (!in_segment) 
            {
                current_start = i;
                in_segment = true;
            }
            size_t current_length = i - current_start + 1;
            if (current_length > max_length) 
            {
                max_length = current_length;
                max_start = current_start;
            }
        }else{
            in_segment = false;
        }
    }

    if (max_length == 0) {
        return {};
    }

    //-- 创建最大不能关联片段的indices
    std::vector<size_t> result;
    for (size_t i = max_start; i < max_start + max_length; ++i) 
    {
        result.push_back(i);
    }

    return result;
}



void featureMerger::getMergedClusterIncremental(const std::vector<elementEdge>& vElements, 
    const std::vector<int>& vKFindices,
    const std::vector<KeyFramePtr>& vKeyFrames,
    std::vector<cv::Point3d>& mergedCloud_global,
    std::vector<int>& involved_elements)
{
    assert(vElements.size()==vKFindices.size());
    assert(vKeyFrames.size()>0);

    mergedCloud_global.clear();
    
    //-- 从关键帧里提取出内参
    float fx = vKeyFrames[0]->mFx;
    float fy = vKeyFrames[0]->mFy;
    float cx = vKeyFrames[0]->mCx;
    float cy = vKeyFrames[0]->mCy;

    // * STEP-1 保存这个 cluster 的边缘以及每个边缘的位姿，并顺便找到最长的边缘
    std::vector<Edge> vEdges;
    std::vector<Sophus::SE3d> vKFposes;
    int N = vElements.size();
    vEdges.reserve(N);
    vKFposes.reserve(N);

    int longest_idx = -1;
    int longest_size = 0;
    for(size_t i = 0; i < N; ++i)
    {
        int ele_edge_idx = vElements[i].kf_edge_idx;
        Edge& edge = vKeyFrames[vKFindices[i]]->mvEdges[ele_edge_idx];
        vEdges.push_back(edge);
        if(edge.mvPoints.size() > longest_size)
        {
            longest_idx = vEdges.size() - 1;
            longest_size = edge.mvPoints.size();
        }

        Sophus::SE3d pose = vKeyFrames[vKFindices[i]]->KF_pose_g;
        vKFposes.push_back(pose);
    }

    // * STEP-2 对最长的边缘进行采样
    std::vector<cv::Point2d> sampled_longest_edge;
    std::vector<Eigen::Vector3d> sampled_longest_edge_3d;
    const int sampleBias = 3;
    Edge& longestEdge = vEdges[longest_idx]; 
    Sophus::SE3d pose_ref = vKFposes[longest_idx];
    
    auto pt2cvPoint2d = [](const orderedEdgePoint& pt) {
        return cv::Point2d(pt.x, pt.y);
    };

    auto pt2Eigen = [](const orderedEdgePoint& pt) {
        return Eigen::Vector3d(pt.x_3d, pt.y_3d, pt.z_3d);
    };

    sampled_longest_edge.reserve(longest_size / sampleBias + 2);
    sampled_longest_edge_3d.reserve(longest_size / sampleBias + 2);

    sampled_longest_edge.push_back(pt2cvPoint2d(longestEdge.mvPoints.front()));  // 第一个点
    sampled_longest_edge_3d.push_back(pt2Eigen(longestEdge.mvPoints.front()));
    for(size_t i = sampleBias; i < longest_size - 1; i += sampleBias) 
    {
        sampled_longest_edge.push_back(pt2cvPoint2d(longestEdge.mvPoints[i]));
        sampled_longest_edge_3d.push_back(pt2Eigen(longestEdge.mvPoints[i]));
    }
    if(longest_size > 1) {  // 最后一个点
        sampled_longest_edge.push_back(pt2cvPoint2d(longestEdge.mvPoints.back()));
        sampled_longest_edge_3d.push_back(pt2Eigen(longestEdge.mvPoints.back()));
    }

    // * STEP-3 其他边缘投影到最长边缘的坐标系下以搜索与之关联的参考边缘坐标点
    size_t N_longest_pts = sampled_longest_edge.size();

    std::vector<std::vector<Eigen::Vector3d>> point_neighbors_total;
    std::vector<std::vector<float>> point_neighbors_score_total;

    point_neighbors_total.resize(N_longest_pts);
    point_neighbors_score_total.resize(N_longest_pts);

    //-- 统计每个 elementEdge 构成 merged cloud 的比例
    std::vector<int> statistics(N, 0);

    //-- 每条边的各个点被参考边缘关联的状态
    std::vector<std::vector<bool>> associatedList_total;
    associatedList_total.resize(N);

    for (size_t i = 0; i < N; ++i)
    {
        if (i == longest_idx) continue;
        Edge& edge = vEdges[i];
        Sophus::SE3d pose_cur = vKFposes[i];
        Sophus::SE3d pose_ref_cur = pose_ref.inverse() * pose_cur;

        int N_edge_pts = edge.mvPoints.size();
        std::vector<bool> isAssociated(N_edge_pts, false);

        for(size_t j = 0; j < N_edge_pts; ++j)
        {
            orderedEdgePoint& ordered_pt = edge.mvPoints[j];
            double score_depth = ordered_pt.score_depth;
            double x_cur_3d = ordered_pt.x_3d;
            double y_cur_3d = ordered_pt.y_3d;
            double z_cur_3d = ordered_pt.z_3d;
            Eigen::Vector3d pt_cur_3d(x_cur_3d, y_cur_3d, z_cur_3d);

            //-- 投影到参考边缘
            Eigen::Vector3d pt_ref_3d = pose_ref_cur * pt_cur_3d;
            double x = fx * pt_ref_3d.x() / pt_ref_3d.z() + cx;
            double y = fy * pt_ref_3d.y() / pt_ref_3d.z() + cy;

            //-- 找到最近的参考边缘点
            int closet_idx = -1;
            double closet_dist = 1000000.0;
            for(size_t k = 0; k < N_longest_pts; ++k)
            {
                double x_cur = sampled_longest_edge[k].x;
                double y_cur = sampled_longest_edge[k].y;
                double dist_2 = (x-x_cur)*(x-x_cur) + (y-y_cur)*(y-y_cur);
                if(dist_2 < closet_dist)
                {
                    closet_dist = dist_2;
                    closet_idx = k;
                }
            }

            if(closet_dist > 5){
                //-- 此时周围边缘点 j 与参考边缘无关联
                continue;
            }

            int left_idx = closet_idx-1 < 0 ? 0 : closet_idx-1;
            int right_idx = closet_idx+1 >= N_longest_pts ? N_longest_pts-1 : closet_idx+1;
            auto [foot, is_between] = calculateFootAndCheck(sampled_longest_edge[left_idx], 
                                                            sampled_longest_edge[right_idx], 
                                                            cv::Point2d(x,y));
            
            if (is_between == false){
                //-- 此时周围边缘点 j 与参考边缘无关联
                continue;
            }

            //-- 深度错位清除
            double z_warp = pt_ref_3d.z();
            double z_ref = sampled_longest_edge_3d[i].z();
            // if(std::abs(z_warp - z_ref) > 0.03)
            // {
            //     continue;
            // }

            auto distance = [](const cv::Point2d& p1, const cv::Point2d& p2) {
                double dx = p2.x - p1.x;
                double dy = p2.y - p1.y;
                return std::sqrt(dx*dx + dy*dy);
            };

            double dist_left = distance(foot, sampled_longest_edge[left_idx]);
            double dist_closet = distance(foot, sampled_longest_edge[closet_idx]);
            double dist_right = distance(foot, sampled_longest_edge[right_idx]);

            //-- 此时方可以确认周围边缘的点 j 与该参考边缘点关联, index 为 asso_index;
            int asso_index;
            if(dist_left <= dist_closet && dist_left < dist_right){
                asso_index = left_idx;
            }else if(dist_closet <= dist_left && dist_closet < dist_right){
                asso_index = closet_idx;
            }else{
                asso_index = right_idx;
            }
            isAssociated[j] = true;

            //-- 为 asso_index 对应的参考边缘点更新邻居
            Eigen::Vector3d pt_global_3d = pose_ref * pt_ref_3d;
            point_neighbors_total[asso_index].push_back(pt_global_3d);
            point_neighbors_score_total[asso_index].push_back(score_depth);

            statistics[i] += 1;

        }

        associatedList_total[i] = isAssociated;
    }

    //-- 将采样的参考边缘变换到全局坐标系下
    for (size_t i = 0; i < sampled_longest_edge_3d.size(); ++i)
    {
        sampled_longest_edge_3d[i] = pose_ref * sampled_longest_edge_3d[i];
    }

    // * STEP-4 计算质心（全局坐标系下）
    std::vector<cv::Point3d> centroid_global;
    for(size_t i = 0; i < N_longest_pts; ++i)
    {
        std::vector<Eigen::Vector3d> neighbors = point_neighbors_total[i];
        std::vector<float> scores = point_neighbors_score_total[i];
        Eigen::Vector3d current = sampled_longest_edge_3d[i];
        
        std::vector<Eigen::Vector3d> cluster = neighbors;
        cluster.push_back(current);
        scores.push_back(1);

        statisticFilter(cluster, scores, 1);

        Eigen::Vector3d centroid(0.0, 0.0, 0.0);

        float sum_total = 0.0f;

        for (size_t j = 0; j < cluster.size(); ++j)
        {
            centroid.x() += cluster[j].x() * scores[j];
            centroid.y() += cluster[j].y() * scores[j];
            centroid.z() += cluster[j].z() * scores[j];
            sum_total += scores[j];
        }
        
        int numPoints = cluster.size();
        if(numPoints <= 1)
        {
            //centroid_global.push_back(NaN);
            continue;
        }
        centroid.x() /= sum_total;
        centroid.y() /= sum_total;
        centroid.z() /= sum_total;
        cv::Point3d p3d_cv(centroid.x(), centroid.y(), centroid.z());
        centroid_global.push_back(p3d_cv);
    }
    if(!centroid_global.empty())
    {
        mergedCloud_global = std::move(centroid_global);
    }

    // * STEP-5 整理构成关系
    involved_elements.clear();
    involved_elements.reserve(N);
    //-- 参考边缘肯定构成
    involved_elements.push_back(longest_idx);
    for (size_t i = 0; i < statistics.size(); ++i)
    {
        if(statistics[i] > 5 || statistics[i] >= 0.5 * vEdges[i].mvPoints.size())
        {
            involved_elements.push_back(i);
        }
    }
}


void featureMerger::assoSurround2Ref(std::vector<Edge>& vEdges,
                      const std::vector<Sophus::SE3d>& vKFposes,
                      size_t ref_edge_index, Sophus::SE3d pose_ref,
                      float fx, float fy, float cx, float cy,
                      const std::vector<cv::Point2d>& ref_edge_sampled_2d,
                      std::vector<std::vector<Eigen::Vector3d>>& point_neighbors_total,
                      std::vector<std::vector<float>>& point_neighbors_score_total,
                      std::vector<std::vector<bool>>& associatedList_total,
                      std::vector<int>& statistics)
{
    size_t N_edges = vEdges.size();
    size_t N_ref_pts = ref_edge_sampled_2d.size();

    point_neighbors_total.resize(N_ref_pts);
    point_neighbors_score_total.resize(N_ref_pts);

    for (size_t i = 0; i < N_edges; ++i)
    {
        Edge& edge = vEdges[i];
        Sophus::SE3d pose_cur = vKFposes[i];
        Sophus::SE3d pose_ref_cur = pose_ref.inverse() * pose_cur;

        int N_edge_pts = edge.mvPoints.size();

        if (i == ref_edge_index)
        {
            //-- 参考边缘就全true
            std::vector<bool> padd(N_edge_pts, true);
            associatedList_total[i] = padd;
            continue;
        }
        associatedList_total[i].resize(N_edge_pts, false);

        for(size_t j = 0; j < N_edge_pts; ++j)
        {
            orderedEdgePoint& ordered_pt = edge.mvPoints[j];
            double score_depth = ordered_pt.score_depth;
            double x_cur_3d = ordered_pt.x_3d;
            double y_cur_3d = ordered_pt.y_3d;
            double z_cur_3d = ordered_pt.z_3d;
            Eigen::Vector3d pt_cur_3d(x_cur_3d, y_cur_3d, z_cur_3d);

            //-- 投影到参考边缘
            Eigen::Vector3d pt_ref_3d = pose_ref_cur * pt_cur_3d;
            double x = fx * pt_ref_3d.x() / pt_ref_3d.z() + cx;
            double y = fy * pt_ref_3d.y() / pt_ref_3d.z() + cy;

            //-- 找到最近的参考边缘点
            int closet_idx = -1;
            double closet_dist = 1000000.0;
            for(size_t k = 0; k < N_ref_pts; ++k)
            {
                double x_cur = ref_edge_sampled_2d[k].x;
                double y_cur = ref_edge_sampled_2d[k].y;
                double dist_2 = (x-x_cur)*(x-x_cur) + (y-y_cur)*(y-y_cur);
                if(dist_2 < closet_dist)
                {
                    closet_dist = dist_2;
                    closet_idx = k;
                }
            }

            if(closet_dist > 5){
                //-- 此时周围边缘点 j 与参考边缘无关联
                continue;
            }

            int left_idx = closet_idx-1 < 0 ? 0 : closet_idx-1;
            int right_idx = closet_idx+1 >= N_ref_pts ? N_ref_pts-1 : closet_idx+1;
            auto [foot, is_between] = calculateFootAndCheck(ref_edge_sampled_2d[left_idx], 
                                                            ref_edge_sampled_2d[right_idx], 
                                                            cv::Point2d(x,y));
            
            if (is_between == false){
                //-- 此时周围边缘点 j 与参考边缘无关联
                continue;
            }

            auto distance = [](const cv::Point2d& p1, const cv::Point2d& p2) {
                double dx = p2.x - p1.x;
                double dy = p2.y - p1.y;
                return std::sqrt(dx*dx + dy*dy);
            };

            double dist_left = distance(foot,   ref_edge_sampled_2d[left_idx]);
            double dist_closet = distance(foot, ref_edge_sampled_2d[closet_idx]);
            double dist_right = distance(foot,  ref_edge_sampled_2d[right_idx]);

            //-- 此时方可以确认周围边缘的点 j 与该参考边缘点关联, index 为 asso_index;
            int asso_index;
            if(dist_left <= dist_closet && dist_left < dist_right){
                asso_index = left_idx;
            }else if(dist_closet <= dist_left && dist_closet < dist_right){
                asso_index = closet_idx;
            }else{
                asso_index = right_idx;
            }
            associatedList_total[i][j] = true;

            //-- 为 asso_index 对应的参考边缘点更新邻居
            Eigen::Vector3d pt_global_3d = pose_ref * pt_ref_3d;
            point_neighbors_total[asso_index].push_back(pt_global_3d);
            point_neighbors_score_total[asso_index].push_back(score_depth);

            statistics[i] += 1;

        }
    }
}

void featureMerger::sampleOEdgeUniform(std::vector<cv::Point2d>& sampled_longest_edge,
                        std::vector<Eigen::Vector3d>& sampled_longest_edge_3d,
                        const std::vector<orderedEdgePoint>& edgePoints, const int sampleBias)
{
    sampled_longest_edge.clear();
    sampled_longest_edge_3d.clear();
    
    auto pt2cvPoint2d = [](const orderedEdgePoint& pt) {
        return cv::Point2d(pt.x, pt.y);
    };

    auto pt2Eigen = [](const orderedEdgePoint& pt) {
        return Eigen::Vector3d(pt.x_3d, pt.y_3d, pt.z_3d);
    };

    int N_points = edgePoints.size();

    sampled_longest_edge.reserve(N_points / sampleBias + 2);
    sampled_longest_edge_3d.reserve(N_points / sampleBias + 2);

    sampled_longest_edge.push_back(pt2cvPoint2d(edgePoints.front()));  // 第一个点
    sampled_longest_edge_3d.push_back(pt2Eigen(edgePoints.front()));
    for(size_t i = sampleBias; i < N_points - 1; i += sampleBias) 
    {
        sampled_longest_edge.push_back(pt2cvPoint2d(edgePoints[i]));
        sampled_longest_edge_3d.push_back(pt2Eigen(edgePoints[i]));
    }
    if(N_points > 1) {  // 最后一个点
        sampled_longest_edge.push_back(pt2cvPoint2d(edgePoints.back()));
        sampled_longest_edge_3d.push_back(pt2Eigen(edgePoints.back()));
    }

}

bool featureMerger::isMergable(const std::vector<Eigen::Vector3d>& ref_seq,
                const std::vector<Eigen::Vector3d>& add_seq,
                MergeType& merge_type,
                double distance_threshold) {
    
    if (ref_seq.empty() || add_seq.empty()) {
        merge_type = CANNOT_MERGE;
        return false;
    }

    // 获取所有端点
    const Eigen::Vector3d& ref_front = ref_seq.front();
    const Eigen::Vector3d& ref_back = ref_seq.back();
    const Eigen::Vector3d& add_front = add_seq.front();
    const Eigen::Vector3d& add_back = add_seq.back();

    // 计算所有端点之间的距离
    std::vector<std::pair<double, std::pair<int, int>>> distances = {
        {(ref_front - add_front).norm(), {0, 0}},
        {(ref_front - add_back).norm(), {0, 1}},
        {(ref_back - add_front).norm(), {1, 0}},
        {(ref_back - add_back).norm(), {1, 1}}
    };

    // 找到最小距离对
    auto min_it = std::min_element(distances.begin(), distances.end(),
        [](const auto& a, const auto& b) { return a.first < b.first; });

    // 如果最小距离大于阈值，不能拼接
    if (min_it->first > distance_threshold) {
        merge_type = CANNOT_MERGE;
        return false;
    }

    // 解引用最小距离对的信息
    int ref_end = min_it->second.first; // 0=front, 1=back of ref_seq
    int add_end = min_it->second.second; // 0=front, 1=back of add_seq

    // 检查ref_seq中离add_seq端点最近的点是否是ref_seq的端点
    const Eigen::Vector3d& add_endpoint = (add_end == 0) ? add_front : add_back;
    double min_dist = std::numeric_limits<double>::max();
    int closest_index = -1;

    for (size_t i = 0; i < ref_seq.size(); ++i) {
        double dist = (ref_seq[i] - add_endpoint).norm();
        if (dist < min_dist) {
            min_dist = dist;
            closest_index = i;
        }
    }

    // 如果最近点不是端点，不能拼接
    if (closest_index != 0 && closest_index != static_cast<int>(ref_seq.size()) - 1) {
        merge_type = CANNOT_MERGE;
        return false;
    }

    // 确定拼接类型
    if (ref_end == 1) { // ref_seq的back与add_seq的某个端点接近
        if (add_end == 0) { // add_seq的front接近ref_seq的back
            merge_type = APPEND_AS_IS;
        } else { // add_seq的back接近ref_seq的back
            merge_type = APPEND_REVERSED;
        }
    } else { // ref_seq的front与add_seq的某个端点接近
        if (add_end == 0) { // add_seq的front接近ref_seq的front
            merge_type = PREPEND_REVERSED;
        } else { // add_seq的back接近ref_seq的front
            merge_type = PREPEND_AS_IS;
        }
    }

    return true;
}

void featureMerger::getMergedClusterIterative(const std::vector<elementEdge>& vElements, 
    const std::vector<int>& vKFindices,
    const std::vector<KeyFramePtr>& vKeyFrames,
    std::vector<cv::Point3d>& mergedCloud_global,
    double& avgDistThres,
    std::vector<int>& involved_elements)
{
    //std::cout<<"**************************************"<<std::endl;
    assert(vElements.size()==vKFindices.size());
    assert(vKeyFrames.size()>0);

    mergedCloud_global.clear();
    
    //-- 从关键帧里提取出内参
    float fx = vKeyFrames[0]->mFx;
    float fy = vKeyFrames[0]->mFy;
    float cx = vKeyFrames[0]->mCx;
    float cy = vKeyFrames[0]->mCy;

    // * STEP-1 保存这个 cluster 的边缘以及每个边缘的位姿，并顺便找到最长的边缘
    std::vector<Edge> vEdges;
    std::vector<Sophus::SE3d> vKFposes;
    int N = vElements.size();
    vEdges.reserve(N);
    vKFposes.reserve(N);

    int longest_idx = -1;
    int longest_size = 0;
    for(size_t i = 0; i < N; ++i)
    {
        int ele_edge_idx = vElements[i].kf_edge_idx;
        Edge& edge = vKeyFrames[vKFindices[i]]->mvEdges[ele_edge_idx];
        vEdges.push_back(edge);
        if(edge.mvPoints.size() > longest_size)
        {
            longest_idx = vEdges.size() - 1;
            longest_size = edge.mvPoints.size();
        }

        Sophus::SE3d pose = vKeyFrames[vKFindices[i]]->KF_pose_g;
        vKFposes.push_back(pose);
    }

    // * STEP-2 对最长的边缘进行采样
    std::vector<cv::Point2d> sampled_longest_edge;
    std::vector<Eigen::Vector3d> sampled_longest_edge_3d;

    const int sampleBias = 3;
    Edge& longestEdge = vEdges[longest_idx]; 
    Sophus::SE3d pose_ref = vKFposes[longest_idx];
    
    sampleOEdgeUniform(sampled_longest_edge, sampled_longest_edge_3d, longestEdge.mvPoints, sampleBias);

    
    
    // * STEP-3-1 其他边缘投影到最长边缘的坐标系下以搜索与之关联的参考边缘坐标点

    //-- 统计每个 elementEdge 构成 merged cloud 的比例
    std::vector<int> statistics(N, 0);

    //-- 每条边的各个点被参考边缘关联的状态
    std::vector<std::vector<bool>> associatedList_total;
    associatedList_total.resize(N);

    std::vector<std::vector<Eigen::Vector3d>> point_neighbors_total;
    std::vector<std::vector<float>> point_neighbors_score_total;

    assoSurround2Ref(vEdges, vKFposes, longest_idx, pose_ref, fx, fy, cx, cy,
        sampled_longest_edge, point_neighbors_total, point_neighbors_score_total, 
        associatedList_total, statistics);

    // * STEP 3-2 拼接无法被第一条参考边缘关联的周围边缘片段
    while(true)
    {
        int max_outlier_size = 0;
        int max_outlier_edge_idx = -1;
        std::vector<size_t> max_outlier_indices;
        for(size_t i = 0; i < N; ++i)
        {
            std::vector<size_t> tmp = findLongestFalseSegment(associatedList_total[i]);
            if(tmp.size() > max_outlier_size)
            {
                max_outlier_size = tmp.size();
                max_outlier_indices = tmp;
                max_outlier_edge_idx = i;
            }
        }
        //std::cout<<max_outlier_size<<std::endl;

        if(max_outlier_size < 3) break;
        
        //-- 获得需要被再次拟合的边缘片段
        Edge& edge = vEdges.at(max_outlier_edge_idx);
        std::vector<orderedEdgePoint> oedge_points;
        for(size_t i = 0; i < max_outlier_indices.size(); ++i)
        {
            size_t index_pt = max_outlier_indices[i];
            oedge_points.push_back(edge.mvPoints.at(index_pt));
        }

        //-- 采样
        Sophus::SE3d pose_additional = vKFposes[max_outlier_edge_idx];
        Sophus::SE3d pose_ref_additional = pose_ref.inverse() * pose_additional;

        std::vector<cv::Point2d> additional_edge_pts;
        std::vector<Eigen::Vector3d> additional_edge_pts_3d;
        sampleOEdgeUniform(additional_edge_pts, additional_edge_pts_3d, oedge_points, sampleBias);
        
        //-- 3D点投影到第一个参考边缘的坐标系下
        for(size_t i = 0; i < additional_edge_pts_3d.size(); ++i)
        {
            additional_edge_pts_3d[i] = pose_ref_additional * additional_edge_pts_3d[i];
        }


        std::vector<std::vector<Eigen::Vector3d>> point_neighbors;
        std::vector<std::vector<float>> point_neighbors_scores;

        assoSurround2Ref(vEdges, vKFposes, max_outlier_edge_idx, pose_additional, fx, fy, cx, cy,
                        additional_edge_pts, point_neighbors, point_neighbors_scores, 
                        associatedList_total, statistics);
        
        MergeType type;
        if(isMergable(sampled_longest_edge_3d, additional_edge_pts_3d, type, 0.05))
        {
            //std::cout<<"merge"<<std::endl;
            mergeVectors(sampled_longest_edge_3d, additional_edge_pts_3d, type);
            mergeVectors(sampled_longest_edge, additional_edge_pts, type);
            mergeVectors(point_neighbors_total, point_neighbors, type);
            mergeVectors(point_neighbors_score_total, point_neighbors_scores, type);

        }
    }

    //-- 将采样的参考边缘变换到全局坐标系下
    for (size_t i = 0; i < sampled_longest_edge_3d.size(); ++i)
    {
        sampled_longest_edge_3d[i] = pose_ref * sampled_longest_edge_3d[i];
    }

    // * STEP-4 计算质心（全局坐标系下）
    assert(sampled_longest_edge.size() == point_neighbors_total.size() &&
           point_neighbors_score_total.size() == point_neighbors_total.size());

    std::vector<cv::Point3d> centroid_global;
    std::vector<double> list_sts_fil_thres;
    for(size_t i = 0; i < sampled_longest_edge.size(); ++i)
    {
        std::vector<Eigen::Vector3d> neighbors = point_neighbors_total[i];
        std::vector<float> scores = point_neighbors_score_total[i];
        Eigen::Vector3d current = sampled_longest_edge_3d[i];
        
        std::vector<Eigen::Vector3d> cluster = neighbors;
        cluster.push_back(current);
        scores.push_back(1);

        double thres_dist;
        thres_dist = statisticFilter(cluster, scores, 1);

        Eigen::Vector3d centroid(0.0, 0.0, 0.0);

        float sum_total = 0.0f;

        for (size_t j = 0; j < cluster.size(); ++j)
        {
            centroid.x() += cluster[j].x() * scores[j];
            centroid.y() += cluster[j].y() * scores[j];
            centroid.z() += cluster[j].z() * scores[j];
            sum_total += scores[j];
        }
        
        int numPoints = cluster.size();
        if(numPoints <= 3)
        {
            //centroid_global.push_back(NaN);
            continue;
        }
        centroid.x() /= sum_total;
        centroid.y() /= sum_total;
        centroid.z() /= sum_total;
        cv::Point3d p3d_cv(centroid.x(), centroid.y(), centroid.z());
        centroid_global.push_back(p3d_cv);
        list_sts_fil_thres.push_back(thres_dist);
    }
    if(!centroid_global.empty())
    {
        mergedCloud_global = std::move(centroid_global);
        double avg = std::accumulate(list_sts_fil_thres.begin(), list_sts_fil_thres.end(), 0.0)/double(list_sts_fil_thres.size());
        avgDistThres = avg;
    }

    // * STEP-5 整理构成关系
    involved_elements.clear();
    involved_elements.reserve(N);
    //-- 参考边缘肯定构成
    involved_elements.push_back(longest_idx);
    for (size_t i = 0; i < statistics.size(); ++i)
    {
        if(statistics[i] > 5 || statistics[i] >= 0.5 * vEdges[i].mvPoints.size())
        {
            involved_elements.push_back(i);
        }
    }
}


