#include "Frame.h"

Frame::Frame(int ID, std::vector<Edge> vEdges, const cv::Mat& matRGB, const cv::Mat& matDepth,
    const float& fx, const float& fy, const float& cx, const float& cy)
{
    //-- 设置关键帧ID
    F_ID = ID;

    //-- 设置内参
    mCx = cx;
    mCy = cy;
    mFx = fx;
    mFy = fy;

    //-- 帧的原始图像的赋值
    cv::cvtColor(matRGB, mMatGray, cv::COLOR_BGR2GRAY);
    mHeight = mMatGray.rows;
    mWidth =  mMatGray.cols;

    mvEdges = std::move(vEdges);

    //-- Depth 有关的预处理，剔除不一致的边缘特征
    assignProperty3D(matDepth); //-- 为边缘赋值深度
    // edgeCullingDepth();         //-- 剔除所有深度无效的边缘以及有效边缘中的无效边缘点
    edgeCullingDepthParallel();
    //-- 现在剩下的edge中的所有edge point 都有有效深度
    edgeCullingContinuity();    //-- 确保每条有序边缘的3D点深度连续一致

    //-- 更新这帧里每个边缘点的 frame_edge_ID 和 frame_point_index
    assignPropertyIdx();
    
    //constructSearchPlain();
    constructSearchPlainParallel();
}

void Frame::searchRadius(float x, float y, double radius, std::vector<orderedEdgePoint>& result)
{
    result.clear();

    //-- 定义搜索区域的矩形边界（整数像素坐标）
    int minX = static_cast<int>(std::max(0.0, x - radius));
    int maxX = static_cast<int>(std::min(mMatSearch.cols - 1.0, x + radius));
    int minY = static_cast<int>(std::max(0.0, y - radius));
    int maxY = static_cast<int>(std::min(mMatSearch.rows - 1.0, y + radius));

    //-- 搜索到的点距 (x,y) 的距离
    std::vector<float> list_distance;

    //-- 遍历搜索区域内的所有像素, 寻找半径内的非（-1，-1）的边缘点
    for(int py = minY; py <= maxY; ++py)
    {
        for(int px = minX; px <= maxX; ++px)
        {
            const cv::Vec2i& pixel = mMatSearch.at<cv::Vec2i>(py, px);
            int edgeID = pixel[0];     //-- frame_edge_ID
            int pointIdx = pixel[1];   //-- frame_point_index

            //-- 跳过无效点
            if (edgeID == -1 || pointIdx == -1) continue;

            //-- 计算距离（欧几里得距离）
            float dx = px - x;
            float dy = py - y;
            float distance = std::sqrt(dx * dx + dy * dy);

            //-- 如果距离在半径内，添加到结果
            if (distance <= radius)
            {
                //-- 获取原始点数据
                const auto& edge = mvEdges[mmIndexMap.at(edgeID)];
                orderedEdgePoint point = edge.mvPoints[pointIdx];

                //-- 确保原始点数据的frame_edge_ID 和 frame_point_index 确实是搜到的结果
                assert(point.frame_edge_ID == edgeID && point.frame_point_index == pointIdx);

                result.push_back(point);
                float distance = sqrt((point.x-x)*(point.x-x) + (point.y-y)*(point.y-y));
                list_distance.push_back(distance);
            }
        }
    }

    assert(list_distance.size() == result.size());

    // 创建索引数组
    std::vector<size_t> indices(result.size());
    std::iota(indices.begin(), indices.end(), 0);

    // 根据相邻点相对(x,y)的距离对索引数组排序
    std::sort(indices.begin(), indices.end(), 
              [&list_distance](size_t i, size_t j) { return list_distance[i] < list_distance[j]; });

    // 根据排序后的索引重新排列 result
    std::vector<orderedEdgePoint> sorted_result;
    sorted_result.reserve(result.size()); 
    for (size_t i : indices) {
        sorted_result.push_back(result[i]);
    }
    result = std::move(sorted_result);
}

bool isPointsAssociated(orderedEdgePoint pt1, orderedEdgePoint pt2)
{
    float res = fabs(pt1.imgGradAngle-pt2.imgGradAngle);
    if(res > 180) res = 360 - res;
    //-- 梯度方向一致性关联
    if(res<10.0){
        return true;
    }else{
        return false;
    }
}

std::vector<int> Frame::edgeWiseCorrespondenceReproject(Edge& query_edge, const Sophus::SE3d& T2curr)
{

    // * STEP 1. 得到参考帧边缘重投影到当前帧的坐标
    std::vector<orderedEdgePoint>& queryList = query_edge.mvPoints;
    std::vector<cv::Point> warped_queryList;
    const size_t num_points = queryList.size();
    warped_queryList.reserve(num_points);

    // 预计算相机内参倒数（减少除法运算）
    const float inv_fx = 1.0f / mFx;
    const float inv_fy = 1.0f / mFy;

    for(int i = 0; i < num_points; ++i)
    {
        //-- 由像素与深度值恢复的3D点
        const auto& pt = queryList[i];
        float z = pt.depth;
        float x = (static_cast<float>(pt.x) - mCx) * inv_fx * z;
        float y = (static_cast<float>(pt.y) - mCy) * inv_fy * z;
        
        //-- 重投影得到新的投影点
        Eigen::Vector3d point = T2curr * Eigen::Vector3d(x, y, z);
        warped_queryList.emplace_back(
            mFx * point.x() / point.z() + mCx,
            mFy * point.y() / point.z() + mCy
        );
    }

    // * STEP 2. 半径邻域搜索，并投票得到 query edge 的每个点最想关联的边缘

    //-- first:当前帧的边的ID   second: 该条当前帧边有几个query edge的点意愿关联
    std::map<int, int> edgeVoteMapTotal;
    const float radius = 6.0f;
    const int threshold_value = std::min(static_cast<int>(num_points * 0.3f), 5);

    for(int i = 0; i < num_points; ++i)
    {
        orderedEdgePoint& pt = queryList[i];
        float x = warped_queryList[i].x;
        float y = warped_queryList[i].y;
        std::vector<orderedEdgePoint> neighbors_points;
        searchRadius(x, y, radius, neighbors_points);

        //-- 预存该点的近邻匹配关系
        pt.mvAssoFrameEdgeIDs.clear();
        pt.mvAssoFramePointIndices.clear();
        pt.mvAssoFrameEdgeIDs.reserve(neighbors_points.size());
        pt.mvAssoFramePointIndices.reserve(neighbors_points.size());

        //-- 对于一个点，建立一个投票，得到这个点最倾向关联的边
        std::unordered_map<int, int> edgeVoteMap;
        for (const auto& neighbor : neighbors_points) 
        {
            if (isPointsAssociated(pt, neighbor)) 
            {
                //-- 直接递增，避免find检查
                edgeVoteMap[neighbor.frame_edge_ID]++;
                //-- 确认可以关联后，更新关联的缓存
                pt.mvAssoFrameEdgeIDs.push_back(neighbor.frame_edge_ID);
                pt.mvAssoFramePointIndices.push_back(neighbor.frame_point_index);
            }
        }

        if (!edgeVoteMap.empty()) 
        {
            // 找出票数最多的边，此时max_pair.first 就是当前 query point 最想关联的边缘
            const auto max_pair = *std::max_element(
                edgeVoteMap.begin(), edgeVoteMap.end(),
                [](const auto& a, const auto& b) { return a.second < b.second; }
            );
            // 每个点只有一个最想关联的边缘
            edgeVoteMapTotal[max_pair.first] += 1;
        }
    }

    // * STEP 3. 整理投票，确认当前query edge 能与哪些 current edges 关联
    //-- 现在得到的edgeVoteMapTotal包含了query edge与 candidate edge关联的投票关系
    
    std::vector<int> result;
    result.reserve(edgeVoteMapTotal.size());  // 预分配内存
    
    //-- 找出满足阈值要求的当前帧可关联边缘
    for (const auto& [edge_id, votes] : edgeVoteMapTotal) 
    {
        if(votes > threshold_value) result.push_back(edge_id);
    }

    if (result.empty()) {
        return result;
    }

    // * STEP 4: 更新关联关系（使用哈希表加速查找）
    const std::unordered_set<int> validAssociation(result.begin(), result.end());
    for(auto& pt : query_edge.mvPoints) 
    {
        // 直接修改原数据，避免拷贝
        for(size_t j = 0; j < pt.mvAssoFrameEdgeIDs.size(); ++j) 
        {
            if(validAssociation.count(pt.mvAssoFrameEdgeIDs[j])) 
            {
                pt.asso_edge_ID = pt.mvAssoFrameEdgeIDs[j];
                pt.asso_point_index = pt.mvAssoFramePointIndices[j];
                pt.mbAssociated = true;
                break;
            }
        }
        // 清空内存（使用swap确保内存释放）
        std::vector<int>().swap(pt.mvAssoFrameEdgeIDs);
        std::vector<int>().swap(pt.mvAssoFramePointIndices);
    }
    
    return result;
}

void Frame::assignPropertyIdx()
{
    //-- 根据edges的ID构造ID与索引的映射
    for(size_t i = 0; i < mvEdges.size(); ++i)
    {
        //-- 更新edge_id与edge在mvEdges中的index的映射关系
        const int edge_id = mvEdges[i].edge_ID;

        if(mmIndexMap.find(edge_id) != mmIndexMap.end())
        {
            std::cout<<"\033[31m"<<"[ERROR]"<<"\033[0m"<<
            " WRONG EDGE POINT INDEX "<<edge_id<<", INDICES SHOULD BE DIFFERENT!"<<std::endl;
            continue;
        }else{
            mmIndexMap[edge_id] = i;
        }

        auto& edge = mvEdges[i];

        //-- 对于边缘中的每个边缘点，更新其对帧中所有边缘的索引
        for(int j = 0; j < edge.mvPoints.size(); ++j)
        {
            auto& point = edge.mvPoints[j];
            //-- 更新边缘id索引
            point.frame_edge_ID = edge_id;
            //-- 更新边缘点列表索引
            point.frame_point_index = static_cast<int>(j);
        }
    }
}

//-- 构建搜索阵列，把所有的 mvEdges 里的所有点怼到一个 cv::Mat 里
void Frame::constructSearchPlain()
{
    // 创建一个 CV_32SC2 类型的 Mat，初始值设为 (-1, -1) 表示无效位置
    mMatSearch = cv::Mat(mHeight, mWidth, CV_32SC2, cv::Scalar(-1, -1));

    for (size_t i = 0; i < mvEdges.size(); ++i) 
    {
        const auto& edge = mvEdges[i];
        for (size_t j = 0; j < edge.mvPoints.size(); ++j) 
        {
            const auto& point = edge.mvPoints[j];
            
            // 确保坐标在图像范围内
            if(point.x >= 0 && point.x < mWidth && point.y >= 0 && point.y < mHeight){
                // 访问指定位置并赋值
                auto& pixel = mMatSearch.at<cv::Vec2i>(point.y, point.x);
                pixel[0] = point.frame_edge_ID;      // 第一个通道存储 edge ID
                pixel[1] = point.frame_point_index;  // 第二个通道存储 point index
            }else{
                std::cerr << "Point (" << point.x << ", " << point.y 
                          << ") out of bounds!" << std::endl;
            }
        }
    }
}

void Frame::constructSearchPlainParallel()
{
    //-- 创建并初始化矩阵
    mMatSearch = cv::Mat(mHeight, mWidth, CV_32SC2, cv::Scalar(-1, -1));
    
    // 使用 parallel_for_each 并行处理所有边
    tbb::parallel_for_each(mvEdges.begin(), mvEdges.end(),
        [&](const auto& edge) {
            // 遍历当前边的所有点
            for (const auto& point : edge.mvPoints) {
                // 直接写入矩阵
                auto& pixel = mMatSearch.at<cv::Vec2i>(point.y, point.x);
                pixel[0] = point.frame_edge_ID;      // 存储 edge ID
                pixel[1] = point.frame_point_index; // 存储 point index
            }
        });
}

cv::Mat Frame::visualizeSearchPlain()
{
    std::map<int, std::vector<cv::Point>> edgeMap;

    for (int y = 0; y < mMatSearch.rows; ++y) {
        for (int x = 0; x < mMatSearch.cols; ++x) {
            const cv::Vec2i& pixel = mMatSearch.at<cv::Vec2i>(y, x);
            int edgeID = pixel[0];  // 通道1: frame_edge_ID
            if (edgeID != -1) {     // 忽略 (-1,-1) 的无效点
                edgeMap[edgeID].emplace_back(x, y);
            }
        }
    }
    // Step 2: 创建彩色图像 (3通道 BGR)
    cv::Mat colorMat(mMatSearch.size(), CV_8UC3, cv::Scalar(0, 0, 0)); // 默认黑色

    // Step 3: 为每个 edgeID 生成随机颜色
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(50, 255); // 避免太暗的颜色

    for (const auto& [edgeID, points] : edgeMap) {
        cv::Scalar color(dis(gen), dis(gen), dis(gen)); // 随机 BGR 颜色

        // 绘制该 edgeID 的所有点
        for (const auto& pt : points) {
            colorMat.at<cv::Vec3b>(pt.y, pt.x) = cv::Vec3b(
                static_cast<uchar>(color[0]),
                static_cast<uchar>(color[1]),
                static_cast<uchar>(color[2])
            );
        }
    }

    return colorMat;
}

void Frame::assignProperty3D(const cv::Mat& matDepth)
{
    // 并行处理外循环
    tbb::parallel_for(0, (int)mvEdges.size(), [&](int i) {
        // 内循环保持串行
        for(int j = 0; j < mvEdges[i].mvPoints.size(); ++j) {
            assignProperty3DEach(mvEdges[i].mvPoints[j], matDepth);
        }
    });

    // for(int i = 0; i < mvEdges.size(); ++i)
    // {
    //     for(int j = 0; j < mvEdges[i].mvPoints.size(); ++j)
    //     {
    //         assignProperty3DEach(mvEdges[i].mvPoints[j], matDepth);
    //     }
    // }
}

void Frame::assignProperty3DEach(orderedEdgePoint& pt, const cv::Mat& matDepth)
{
    int x_idx = pt.x;
    int y_idx = pt.y;

    //-- 原本点的真实深度
    float depth_orig = matDepth.at<float>(y_idx, x_idx);

    //-- 在5x5的patch中计算修正深度以及可见性分数
    std::vector<float> validDepthList; //-- 所有深度不为0的点的深度列表
    int patch_total = 0;               //-- 当前的patch一共有多少像素
    for(int x_bias = -2; x_bias <= 2; ++x_bias){
        for(int y_bias = -2; y_bias <= 2; ++y_bias){
            int curr_x_idx = x_idx + x_bias;
            int curr_y_idx = y_idx + y_bias;
            
            //-- 判断该位置在不在图像区域里
            if(curr_x_idx < 0 || curr_x_idx >= mWidth ||
               curr_y_idx < 0 || curr_y_idx >= mHeight) continue;

            patch_total += 1; //-- 累积总体像素
            //-- 在图像区域里的话判断深度值是否有效
            float depth = matDepth.at<float>(curr_y_idx, curr_x_idx);
            if(depth > 0.2) validDepthList.push_back(depth);

        }
    }
    std::sort(validDepthList.begin(), validDepthList.end());//-- 从小到大排序
    int size = validDepthList.size();
    float adjusted_depth = 0;     //-- 矫正后的深度

    //-- 计算前景深度
    if(size >= 8){
        //-- 检查跳变并返回第一组连续数据
        std::vector<size_t> jump_indices;
        float rel_thres = 0.05;
        for (size_t i = 1; i < validDepthList.size(); ++i){
            float dx = validDepthList[i] - validDepthList[i-1];
            float x = validDepthList[i-1];
            float relative_change = dx/x; // validDepthList 中均为大于0的数，不担心除0

            if (std::fabs(relative_change) > rel_thres) {
                jump_indices.push_back(i); // 记录跳变位置
                break;
            }
        }
        
        //-- 如果一个 patch 内深度值有不连续，则抠出最小的那一部分区域
        std::vector<float> adjustDepthList;
        if(jump_indices.empty())
        {
            adjustDepthList = validDepthList;
        }else{
            size_t first_jump = jump_indices[0];
            adjustDepthList = std::vector<float>(validDepthList.begin(), validDepthList.begin() + first_jump);
        }

        int partitionSize = adjustDepthList.size();
        //-- 取最小的部分的深度的中位数作为深度值
        float medianValue = (partitionSize%2==0) ? 
                            (adjustDepthList[partitionSize/2-1] + adjustDepthList[partitionSize/2])/2.0 : 
                            adjustDepthList[partitionSize/2];
        if(depth_orig >= adjustDepthList.front() && depth_orig <= adjustDepthList.back()){
            //-- 如果真实深度在这个区间之间，就取真实深度（真实深度本身是前景）
            adjusted_depth = depth_orig;
        }else{
            //-- 如果真实深度不在前景区间，则修改深度为前景区间
            adjusted_depth = medianValue;
        }
    }

    pt.depth = adjusted_depth; //-- 为点特征的深度进行赋值

    //-- 计算远近分数
    if(pt.depth > 0.2){
        Eigen::Vector3d pt_3d;
        pt_3d.x() = (pt.x - mCx)/mFx * pt.depth;
        pt_3d.y() = (pt.y - mCy)/mFy * pt.depth;
        pt_3d.z() = pt.depth;
        double range = pt_3d.norm();
        //-- 使用反sigmoid函数计算远近分数
        pt.score_depth = 1.0 / (std::exp((range - 2.5) * 1.0) + 1);
        //-- 更新类中的3D点
        pt.x_3d = pt_3d.x();
        pt.y_3d = pt_3d.y();
        pt.z_3d = pt_3d.z();
    }else{
        pt.score_depth = 0;
    }
    
}

//-- 剔除所有深度无效的边缘以及有效边缘中的无效边缘点
void Frame::edgeCullingDepth()
{
    //-- 遍历所有的边缘，祛除深度大量无效的边缘
    for(auto edgeIter = mvEdges.begin(); edgeIter != mvEdges.end(); )
    {
        //-- 获取当前边缘的引用，避免拷贝
        Edge& currentEdge = *edgeIter;

        int validPointCount = 0;
        int totalPointCount = currentEdge.mvPoints.size();
        
        //-- 先统计有效点的数量
        for(const auto& point : currentEdge.mvPoints)
        {
            if(point.depth > 0.2f && point.depth < 5.0f)
            {
                validPointCount++;
            }
        }

        //-- 计算有效点比例
        float validRatio = static_cast<float>(validPointCount) / totalPointCount;

        if(validRatio >= 0.3f)
        {
            //-- 如果边缘保留，则移除其中的无效点
            auto newEnd = std::remove_if(currentEdge.mvPoints.begin(), 
                                        currentEdge.mvPoints.end(),
                                        [](const auto& point) {
                                            return point.depth <= 0.2f || point.depth >= 5.0f;
                                        });
            currentEdge.mvPoints.erase(newEnd, currentEdge.mvPoints.end());
            edgeIter++;  // 保留这个边缘，移动到下一个
        }
        else
        {
            //-- 如果边缘无效（70%以上都是无效点），则移除整个边缘
            edgeIter = mvEdges.erase(edgeIter);
        }
    }
}

void Frame::edgeCullingDepthParallel()
{
    // 使用 char 代替 atomic<bool>，并用 memory_order_relaxed 保证基本线程安全
    std::vector<char> retainFlags(mvEdges.size());

    tbb::parallel_for(0, (int)mvEdges.size(), [&](int i) {
        Edge& currentEdge = mvEdges[i];
        int validPointCount = 0;
        const int totalPointCount = currentEdge.mvPoints.size();
        
        // 统计有效点数量
        for(const auto& point : currentEdge.mvPoints) {
            if(point.depth > 0.2f && point.depth < 5.0f) {
                validPointCount++;
            }
        }

        // 计算有效比例并决定是否保留
        float validRatio = static_cast<float>(validPointCount) / totalPointCount;
        retainFlags[i] = (validRatio >= 0.3f) ? 1 : 0;

        // 如果是保留的边缘，先过滤掉无效点
        if(retainFlags[i]) {
            auto newEnd = std::remove_if(currentEdge.mvPoints.begin(), 
                                        currentEdge.mvPoints.end(),
                                        [](const auto& point) {
                                            return point.depth <= 0.2f || point.depth >= 5.0f;
                                        });
            currentEdge.mvPoints.erase(newEnd, currentEdge.mvPoints.end());
        }
    });

    // 第二阶段：串行执行实际删除操作
    auto newEnd = std::remove_if(mvEdges.begin(), mvEdges.end(),
        [&retainFlags, &mvEdges = this->mvEdges](const Edge& edge) {
            size_t index = &edge - &mvEdges[0];
            return retainFlags[index] == 0;
        });
    mvEdges.erase(newEnd, mvEdges.end());
}

//-- 确保每条有序边缘的3D点深度连续一致
void Frame::edgeCullingContinuity()
{
    std::vector<bool> isEdgeValid(mvEdges.size(), true);
    //-- 在CullingDepth之后调用，此时认为Edge中每个点都含有有效的深度
    tbb::parallel_for(0, (int)mvEdges.size(), [&](int cnt) {
    //for(int cnt = 0; cnt < mvEdges.size(); ++cnt)
        Edge& edge = mvEdges[cnt];
        
        //* STEP 1. 检索边缘的深度跳变
        std::vector<bool> jumpFlags(edge.mvPoints.size(), false);
        float lastDepth = edge.mvPoints[0].depth;
        for (size_t i = 1; i < edge.mvPoints.size(); ++i) 
        {
            //-- 当前点的深度
            float currentDepth = edge.mvPoints[i].depth;
            //-- 比较深度判断是否连续
            jumpFlags[i] = (std::fabs(currentDepth - lastDepth) > 0.05f);
            lastDepth = currentDepth;
        }
        //-- 跳变次数
        int jump_num = std::count(jumpFlags.begin(), jumpFlags.end(), true);
        float jump_avg =  static_cast<float>(edge.mvPoints.size())/static_cast<float>(jump_num);
        if(jump_avg < 5){
            //-- 如果跳变的比较多，说明该边缘正处于前后景模糊的位置
            isEdgeValid[cnt] = false;
            //-- 对于这样的边缘，考虑直接删而不重新拼,故而跳过
            return;
        }

        //*STEP 2. 对于跳变的不多的边缘，先根据 jumpFlags 进行切片
        int start_ptr = 0;
        //-- 一个 edge 被拆出的 segment, first是首 index，second 是末 index
        std::vector<std::pair<int, int>> segment;

        for(size_t i = 1; i < jumpFlags.size(); ++i)
        {
            if(jumpFlags[i])
            {
                //-- 第 i 个位置跳变了说明 start -- i-1 这一段是连续的
                segment.push_back(std::make_pair(start_ptr, i-1));
                start_ptr = i;
            }
        }

        //-- 最后一截拼入, 此时segment中包含所有的边缘切片（包括不连续的单个点的切片）
        segment.push_back(std::make_pair(start_ptr, jumpFlags.size()-1));
        
        // * STEP 3. 使用并查集合并连续的片段
        //-- 表示并查集
        DisjointSet mergeSet(segment.size());

        for(int i = 0; i < segment.size(); ++i)
        {
            float depth_end = edge.mvPoints[segment[i].second].depth; //-- 片段末端点的深度
            //-- 判断后续的片段能不能和第i段相拼接
            for(int j = i+2; j < segment.size(); ++j)
            {
                float depth_front = edge.mvPoints[segment[j].first].depth; //-- 片段首端点的深度
                //-- 两个片段首末端点深度连续说明可以拼
                if(std::fabs(depth_front - depth_end) < 0.01)
                {
                    //-- 如果能拼上就在并查集上合并这两个节点
                    mergeSet.to_union(i,j);
                }
            }
        }

        // * STEP 4. 挑选最大连续片段以进行后续拼接
        //-- 整理并查集，得到每个集合的总点数
        mergeSet.pruningSet();
        std::map<int, std::vector<int>> cluster; //-- first是root_idx, second是集合中的所有片段的索引
        for(int i = 0; i < segment.size(); ++i)
        {
            int root_idx = mergeSet.find(i);
            cluster[root_idx].push_back(i);
        }
        //-- 此时cluster.second中这些片段本身也是有序的
        int max_length = -1;
        std::vector<int> max_cluster;
        for(const auto& pair : cluster)
        {
            std::vector<int> current_cluster = pair.second;
            int current_length = 0;
            
            //-- 计算当前cluster的长度
            for(int i = 0; i < current_cluster.size(); ++i)
            {
                const auto& seg = segment[current_cluster[i]];
                current_length += seg.second - seg.first + 1;
            }

            if(current_length > max_length){
                max_length = current_length;
                max_cluster = current_cluster;
            }
        }

        // * STEP 5. 根据 max_cluster 拼接
        //-- 重新捏一个mvPoints出来
        std::vector<orderedEdgePoint> new_mvPoints;
        for(int i = 0; i < max_cluster.size(); ++i)
        {
            //-- 每个切片的首尾index
            int start_idx = segment[max_cluster[i]].first;
            int end_index = segment[max_cluster[i]].second;
            for(int j = start_idx; j <= end_index; ++j)
            {
                new_mvPoints.push_back(edge.mvPoints[j]);
            }
        }
        if(new_mvPoints.size() >= 5)
        {
            edge.mvPoints = new_mvPoints;
        }else{
            isEdgeValid[cnt] = false;
        }
    });

    // * STEP 6. 移除拼接完成后过短的边缘以及跳变过多的边缘
    int cnt_idx = 0;
    for(auto iter = mvEdges.begin(); iter != mvEdges.end(); )
    {
        if(isEdgeValid[cnt_idx] == true){
            iter ++;
        }else{
            iter = mvEdges.erase(iter);
        }
        cnt_idx += 1;
    }
}

std::vector<orderedEdgePoint> Frame::getCoarseSampledPoints(int bias, int maximum_point)
{
    std::vector<orderedEdgePoint> selectedPoints;
    for(int i = 0; i < mvEdges.size(); ++i)
    {
        const Edge& edge = mvEdges[i];
        //-- 获取采样的序列
        for(int j = 0; j < edge.mvPoints.size(); ++j)
        {
            const orderedEdgePoint& pt = edge.mvPoints[j];
            selectedPoints.push_back(pt);
        }
    }
    //-- 根据空间均匀的原则进行采样，分数高的点优先
    //-- 对点按分数进行排序
    std::sort(selectedPoints.begin(),selectedPoints.end(),
              [](const orderedEdgePoint& a, const orderedEdgePoint& b){ 
                 return a.score_depth > b.score_depth; 
              });

    //-- 创建全黑的图像作为掩膜
    cv::Mat mask(mHeight, mWidth, CV_8U, cv::Scalar(0));
    //-- 根据掩膜进行采样排序
    std::vector<orderedEdgePoint> sampledPoints;
    sampledPoints.reserve(std::min(maximum_point, static_cast<int>(selectedPoints.size())));

    for(const auto& pt : selectedPoints)
    {
        if(mask.at<uint8_t>(pt.y, pt.x) == 255) continue;
        //-- 当前点可以选择
        sampledPoints.push_back(pt);
        cv::circle(mask, cv::Point(pt.x, pt.y), bias, 255, -1);
        if(sampledPoints.size() >= maximum_point) break;
    }
    //-- 目前是完全采样完成的所有点
    return sampledPoints;
}

void Frame::getFineSampledPoints(int bias)
{
    for(int i = 0; i < mvEdges.size(); ++i)
    {
        Edge& edge = mvEdges[i];
        edge.samplingEdgeUniform(bias);
    }
}






