#include "fineTracker.h"

using namespace fine;

FineTracker::FineTracker(double fx, double fy, double cx, double cy, float ratio): 
mFx(fx), mFy(fy), mCx(cx), mCy(cy), geo_photo_ratio(ratio)
{
    mpF_ref = nullptr;
    mpF_cur = nullptr;
}

void FineTracker::setReference(const FramePtr kf_ref)
{
    if(kf_ref)
    {
        mpF_ref = kf_ref;
    }
}

void FineTracker::setCurrent(const FramePtr f_curr)
{
    if(f_curr)
    {
        mpF_cur = f_curr;
    }
}

//-- 输入当前帧到参考帧的位姿先验
void FineTracker::setPosePriorCur2Ref(const Sophus::SE3d& T)
{
    T_cur_ref = T;
}

//-- 输入参考帧到当前帧的位姿先验
void FineTracker::setPosePriorRef2Cur(const Sophus::SE3d& T)
{
    T_cur_ref = T.inverse();
}


void FineTracker::getAssociationLines2D()
{
    const size_t num_points = mvGeometryPoints.size();
    //-- mvAssociatedLines 初始化成全 NaN, 且与mvGeometryPoints同大小
    mvAssociatedLines.clear();
    mvAssociatedLines.resize(num_points, std::pair<Eigen::Vector2d, Eigen::Vector2d>(
        Eigen::Vector2d(FLT_MAX, FLT_MAX), 
        Eigen::Vector2d(FLT_MAX, FLT_MAX)));

    for(size_t i = 0; i < num_points; ++i)
    {
        const orderedEdgePoint& pt = mvGeometryPoints[i];
        if(!pt.mbAssociated) {
            continue; // Already initialized to NaN
        }

        const auto it = mpF_cur->mmIndexMap.find(pt.asso_edge_ID);
        if(it == mpF_cur->mmIndexMap.end())
        {
            std::cout<<pt.asso_edge_ID<<std::endl;
            std::cout<<"\033[31m [INDEX ERROR] \033[0m"
                    <<": Encounter an edge id that doesn't exist!"<< std::endl;
            continue; // Already initialized to NaN
        }

        const int edge_index = it->second;
        const auto& edge_points = mpF_cur->mvEdges[edge_index].mvPoints;
        const size_t num_edge_points = edge_points.size();

        // Clamp indices to valid range
        const int pt_idx_1 = std::max(pt.asso_point_index - 1, 0);
        const int pt_idx_2 = std::min(pt.asso_point_index + 1, static_cast<int>(num_edge_points) - 1);

        const orderedEdgePoint& pt_1 = edge_points[pt_idx_1];
        const orderedEdgePoint& pt_2 = edge_points[pt_idx_2];
        
        mvAssociatedLines[i] = {
            Eigen::Vector2d(pt_1.x, pt_1.y),
            Eigen::Vector2d(pt_2.x, pt_2.y)
        };
    }
}


void FineTracker::associationRef2Cur()
{
    //-- 为当前帧边缘创建一个bool列表，记录这些边缘是否与参考帧构成关联
    int ref_edge_num = mpF_ref->mvEdges.size();

    //-- 清空几何残差点集
    mvGeometryPoints.clear();

    //-- 用参考帧的有效边缘去当前帧中关当前帧的边缘，记录是否存在关联
    for(int i = 0; i < mpF_ref->mvEdges.size(); ++i)
    {

        Edge& query_edge = mpF_ref->mvEdges[i];

        //-- 关联前要先清空之前的关联点，因为参考帧会多次关联多个当前帧
        for (auto& pt : query_edge.mvPoints)
        {
            pt.mbAssociated = false;
            pt.asso_edge_ID = -1;
            pt.asso_point_index = -1;
            pt.mvAssoFrameEdgeIDs.clear();
            pt.mvAssoFramePointIndices.clear();
        }
        
        //-- 将参考帧的边缘投影到当前帧 进行edge-wise correspondence，
        //-- 正确关联的参考帧边缘点的 asso_edge_ID 和 asso_point_index 会存有其关联的对应当前帧边缘点
        std::vector<int> associated_edges = mpF_cur->edgeWiseCorrespondenceReproject(query_edge, T_cur_ref);
        
        if(associated_edges.empty())
            continue;

        //-- 此时 query_edge 与当前帧存在关联，确定 query_edge 参与优化的采样点
        std::vector<orderedEdgePoint> tmp_pts;
        Edge& edge = mpF_ref->mvEdges[i];
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
        mvGeometryPoints.insert(mvGeometryPoints.end(), tmp_pts.begin(), tmp_pts.end());
    }

    //-- 构造完 mvGeometryPoints 后即构造 mvAssociatedLines 以获得对应的点线关联
    getAssociationLines2D();
}

void FineTracker::associationRef2CurParallel()
{
    //-- 为当前帧边缘创建一个bool列表，记录这些边缘是否与参考帧构成关联
    int ref_edge_num = mpF_ref->mvEdges.size();

    //-- 清空几何残差点集
    mvGeometryPoints.clear();

    // 使用并发容器存储结果
    tbb::concurrent_vector<orderedEdgePoint> concurrent_geometry_points;
    
    tbb::parallel_for(0, (int)mpF_ref->mvEdges.size(), [&](int i) {
        Edge& query_edge = mpF_ref->mvEdges[i];

        //-- 关联前要先清空之前的关联点
        for (auto& pt : query_edge.mvPoints) {
            pt.mbAssociated = false;
            pt.asso_edge_ID = -1;
            pt.asso_point_index = -1;
            pt.mvAssoFrameEdgeIDs.clear();
            pt.mvAssoFramePointIndices.clear();
        }
        
        //-- 将参考帧的边缘投影到当前帧
        std::vector<int> associated_edges = mpF_cur->edgeWiseCorrespondenceReproject(query_edge, T_cur_ref);
        
        if(associated_edges.empty())
            return;

        //-- 此时 query_edge 与当前帧存在关联，确定 query_edge 参与优化的采样点
        Edge& edge = mpF_ref->mvEdges[i];
        
        if(edge.mvSampledEdgeIndex.empty()) {
            edge.samplingEdgeUniform(4);
        }
        
        const auto& index_sampled = edge.mvSampledEdgeIndex;
        for(int j = 0; j < index_sampled.size(); ++j) {
            orderedEdgePoint& pt = edge.mvPoints[index_sampled[j]];
            if (pt.mbAssociated) {
                // 改用push_back逐个添加
                concurrent_geometry_points.push_back(pt);
            }
        }
    });

    // 将并发容器的内容转移到最终结果中
    mvGeometryPoints.reserve(mvGeometryPoints.size() + concurrent_geometry_points.size());
    mvGeometryPoints.insert(mvGeometryPoints.end(), 
                          concurrent_geometry_points.begin(), 
                          concurrent_geometry_points.end());

    //-- 构造完 mvGeometryPoints 后即构造 mvAssociatedLines 以获得对应的点线关联
    getAssociationLines2D();
}

std::pair<Eigen::Vector2d, Eigen::Vector2d> 
normalizeLinePoints2D(Eigen::Vector2d startPointRaw, Eigen::Vector2d endPointRaw)
{
    Eigen::Vector2d dirRaw = endPointRaw - startPointRaw;
    Eigen::Vector2d dirNorm = dirRaw/dirRaw.norm();
    Eigen::Vector2d startPoint = startPointRaw;
    Eigen::Vector2d endPoint = startPoint + dirNorm;
    std::pair<Eigen::Vector2d, Eigen::Vector2d> linePair;
    linePair.first = startPoint; linePair.second = endPoint;
    return linePair;
}

void FineTracker::calculateJacobiPointLine3D2D(Eigen::Vector2d startPoint, 
                                               Eigen::Vector2d endPoint, 
                                               Eigen::Vector3d queryPoint_3d,
                                               Mat66d& H_out, Vec6d& g_out, Eigen::Vector2d& residual)
{
    //-- normalize border points.
    std::pair<Eigen::Vector2d, Eigen::Vector2d> normPoints = normalizeLinePoints2D(startPoint,endPoint);
    
    //-- defination of registrated line.
    Eigen::Vector2d a = normPoints.first;
    Eigen::Vector2d b = normPoints.second;
    Eigen::Vector2d l_ba = a-b;

    Mat66d H = Eigen::Matrix<double , 6, 6>::Zero();
    Vec6d  g = Eigen::Matrix<double , 6, 1>::Zero();

    //-- 计算3D空间的位姿变换
    
    Eigen::Vector3d transed_point = T_cur_ref * queryPoint_3d;
    //-- 计算重投影坐标
    double inv_z = 1.0 / transed_point[2];
    double inv_z2 = inv_z * inv_z;
    Eigen::Vector2d proj(mFx * transed_point[0] / transed_point[2] + mCx, 
                         mFy * transed_point[1] / transed_point[2] + mCy);

    
    //-- 3D位姿变换产生的3D-2D投影的位姿残差的雅克比矩阵为2x6
    Eigen::Matrix<double , 2, 6> J_orig = Eigen::Matrix<double , 2, 6>::Zero();
    Eigen::Matrix<double , 2, 6> J = Eigen::Matrix<double , 2, 6>::Zero();
    //-- 一般重投影误差的2x6残差矩阵, fx/z如果是负的说明这是 关联像素-重投影像素，正的则是 重投影像素-关联像素
    J_orig << -mFx * inv_z,
        0,
        mFx * transed_point[0] * inv_z2,
        mFx * transed_point[0] * transed_point[1] * inv_z2,
        -mFx - mFx * transed_point[0] * transed_point[0] * inv_z2,
        mFx * transed_point[1] * inv_z,
        0,
        -mFy * inv_z,
        mFy * transed_point[1] * inv_z,
        mFy + mFy * transed_point[1] * transed_point[1] * inv_z2,
        -mFy * transed_point[0] * transed_point[1] * inv_z2,
        -mFy * transed_point[0] * inv_z;
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


float FineTracker::GetPixelValue(const cv::Mat &img, float x, float y)
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
void FineTracker::calculateJacobiPhotometric(const cv::Mat& image_ref, const cv::Mat& image_cur,
    cv::Point queryPixel, float depth,
    Mat66d& H_out, Vec6d& g_out, double& cost_out)
{
    //-- 重置所有的矩阵的值，这样返回无效值就都是0
    Mat66d H_bb_curr = Mat66d::Zero(); H_out = H_bb_curr;
    Vec6d  g_b_curr  =  Vec6d::Zero(); g_out = g_b_curr;
    cost_out = 0;

    //-- 光度误差并不是基于单个像素，而是基于整个块, 块大小即half_patch_size
    const int half_patch_size = 1;
    int wl = image_ref.cols, hl = image_ref.rows;

    //-- 使用内参反投影得到参考帧下归一化的相机坐标点
    float ref_norm_x = (queryPixel.x - mCx)/mFx;
    float ref_norm_y = (queryPixel.y - mCy)/mFy;
    Eigen::Vector3d point_ref = Eigen::Vector3d(ref_norm_x, ref_norm_y, 1);
    
    //-- 使用先验深度得到参考帧坐标系下实际的3D点
    point_ref *= depth;
    //-- 利用位姿变换得到当前帧下点的3D坐标
    Eigen::Vector3d point_cur = T_cur_ref * point_ref;
    if(point_cur.z() < 0) return;//-- 深度为负的点不考虑放入优化模型中

    //-- 利用内参投影得到点在当前帧的像素坐标
    float u = mFx * point_cur.x() / point_cur.z() + mCx;
    float v = mFy * point_cur.y() / point_cur.z() + mCy;
    //-- 投影出图像的点不考虑放入优化模型中
    if (u < half_patch_size || u > wl - half_patch_size || 
        v < half_patch_size || v > hl - half_patch_size) return;
    
    double X = point_cur.x(), Y = point_cur.y(), Z = point_cur.z();
    double Z2 = Z * Z, Z_inv = 1.0 / Z, Z2_inv = Z_inv * Z_inv;

    //-- 像素关于李代数的雅克比矩阵，维度为2x6
    Eigen::Matrix<double, 2, 6> J_pixel_xi;
    //-- 具体推导可看slam14讲或是本目录下文档的推导
    J_pixel_xi(0, 0) = mFx * Z_inv;
    J_pixel_xi(0, 1) = 0;
    J_pixel_xi(0, 2) = -mFx * X * Z2_inv;
    J_pixel_xi(0, 3) = -mFx * X * Y * Z2_inv;
    J_pixel_xi(0, 4) = mFx + mFx * X * X * Z2_inv;
    J_pixel_xi(0, 5) = -mFx * Y * Z_inv;

    J_pixel_xi(1, 0) = 0;
    J_pixel_xi(1, 1) = mFy * Z_inv;
    J_pixel_xi(1, 2) = -mFy * Y * Z2_inv;
    J_pixel_xi(1, 3) = -mFy - mFy * Y * Y * Z2_inv;
    J_pixel_xi(1, 4) = mFy * X * Y * Z2_inv;
    J_pixel_xi(1, 5) = mFy * X * Z_inv;

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

void FineTracker::RegistrationCombined()
{
    const cv::Mat& img_ref_gray = mpF_ref->mMatGray;
    const cv::Mat& img_cur_gray = mpF_cur->mMatGray;

    const int iterations = 100;
    int N_1 = mvGeometryPoints.size();

    double last_cost = FLT_MAX;
    double current_cost_geo, current_cost_pho;

    for(int iter=0; iter<iterations; ++iter)
    {
        //-- 分别定义光度误差和几何误差的雅克比矩阵
        current_cost_geo = 0;
        current_cost_pho = 0;

        //-- 统计几何残差与光度残差的增量方程以及残差，以便进行基于分布的鲁棒优化
        std::vector<fine::Mat66d> H_geo_list;
        std::vector<fine::Vec6d>  g_geo_list;
        std::vector<Eigen::Vector2d> residual_geo_list;
        std::vector<float> weight_geo_list; //-- 每个几何残差的分布权重

        std::vector<fine::Mat66d> H_pho_list;
        std::vector<fine::Vec6d>  g_pho_list;
        std::vector<double> cost_pho_list;
        std::vector<double> weight_pho_list; //-- 每个光度残差的分布权重
        
        //-- 更新几何残差的雅克比矩阵
        for(int i=0; i<N_1; ++i)
        {
        	// 传入的是边缘点，转化为Eigen的点
            Eigen::Vector3d p1_i(mvGeometryPoints[i].x_3d, mvGeometryPoints[i].y_3d, mvGeometryPoints[i].z_3d);
            //-- 找到关联的点对
            std::pair<Eigen::Vector2d, Eigen::Vector2d> asso = mvAssociatedLines[i];
            //-- 如果是无效的点则略过不予优化
            if(asso.first.x()==FLT_MAX) continue;
            
            Eigen::Matrix<double , 6, 6> H_curr = Eigen::Matrix<double , 6, 6>::Zero();
            Eigen::Matrix<double , 6, 1> g_curr = Eigen::Matrix<double , 6, 1>::Zero();
            Eigen::Vector2d r_curr;
            calculateJacobiPointLine3D2D(asso.first, asso.second, p1_i, H_curr, g_curr, r_curr);
            double weight = mvGeometryPoints[i].score_depth;
            
            // 累加
            H_geo_list.push_back(H_curr * weight * weight);
            g_geo_list.push_back(g_curr * weight * weight);
            residual_geo_list.push_back(r_curr);

            current_cost_geo += r_curr.squaredNorm();
        }

        //-- 几何残差构建完之后，对几何残差进行卡方检验以进行加权
        eslam_core::robustWeight2D dst_geo(residual_geo_list);
        dst_geo.computeWeights("Huber");
        weight_geo_list = dst_geo.weights;


        //-- 更新光度残差的雅克比矩阵
        for(int i=0; i< N_1; ++i)
        {
            //-- 光度误差只需要2D的点信息以及深度
            cv::Point pt_2d(mvGeometryPoints[i].x, mvGeometryPoints[i].y);
            float depth = mvGeometryPoints[i].depth;
            Eigen::Matrix<double , 6, 6> H_curr = Eigen::Matrix<double , 6, 6>::Zero();
            Eigen::Matrix<double , 6, 1> g_curr = Eigen::Matrix<double , 6, 1>::Zero();
            double cost_curr;
            calculateJacobiPhotometric(img_cur_gray, img_ref_gray, pt_2d, depth,
                                       H_curr, g_curr, cost_curr);
            double weight = mvGeometryPoints[i].score_depth;

            // 累加, 由于光度会有返回0（无效点）的情况，故而这里需要略过这些无效值
            if(g_curr.norm() != 0){
                H_pho_list.push_back(H_curr * weight * weight);
                g_pho_list.push_back(g_curr * weight * weight);
                cost_pho_list.push_back(cost_curr);
                current_cost_pho += cost_curr;
            }
        }

        //-- 光度残差构建完之后，对光度残差进行卡方检验以进行加权
        eslam_core::robustWeightChi2 dst_pho(cost_pho_list); //-- 认为cost_geo_list满足卡方分布
        for(int i = 0; i < H_pho_list.size(); ++i){
            double weight_dist = dst_pho.computeWight(cost_pho_list[i]);
            weight_pho_list.push_back(weight_dist);
        }

        //-- 总的增量方程矩阵
        Eigen::Matrix<double , 6, 6> H = Eigen::Matrix<double , 6, 6>::Zero();
        Eigen::Matrix<double , 6, 1> g = Eigen::Matrix<double , 6, 1>::Zero();

        double geo_pho_ratio = geo_photo_ratio; //-- 几何残差与光度残差比例
        
        //-- 将所有残差块都更新到总的增量方程矩阵中
        for(int i = 0; i < H_geo_list.size(); ++i){
            H += H_geo_list[i] * weight_geo_list[i] * geo_pho_ratio;
            g += g_geo_list[i] * weight_geo_list[i] * geo_pho_ratio;
        }
        
        for(int i = 0; i < H_pho_list.size(); ++i){
            H += H_pho_list[i] * weight_pho_list[i] / double(255 * 255 * 9);
            g += g_pho_list[i] * weight_pho_list[i] / double(255 * 255 * 9);
        }

        H /= double(H_geo_list.size() + H_pho_list.size());
        g /= double(g_geo_list.size() + g_pho_list.size());

        Eigen::Matrix<double , 6, 1> dx = H.ldlt().solve(g);// 求解dx
        if(isnan(dx[0])){
            std::cout << "result is nan"<<std::endl;
        }
        // iter>0用来控制除去第一次，因为初始current_cost、last_cost都是0
        if(iter > 0 && current_cost_geo + current_cost_pho > last_cost){
            break;
        }
        // 进行更新，这里的dx是李代数
        T_cur_ref = Sophus::SE3d::exp(dx) * T_cur_ref;
        last_cost = current_cost_geo + current_cost_pho ;

        if(dx.norm() < 1e-6){
            break;
        }
    }
}

void FineTracker::RegistrationCombinedParallel()
{
    const cv::Mat& img_ref_gray = mpF_ref->mMatGray;
    const cv::Mat& img_cur_gray = mpF_cur->mMatGray;

    const int iterations = 100;
    int N_1 = mvGeometryPoints.size();

    double last_cost = FLT_MAX;
    double current_cost_geo, current_cost_pho;

    for(int iter=0; iter<iterations; ++iter)
    {
        //-- 分别定义光度误差和几何误差的雅克比矩阵
        current_cost_geo = 0;
        current_cost_pho = 0;

        //-- 统计几何残差与光度残差的增量方程以及残差，以便进行基于分布的鲁棒优化
        std::vector<fine::Mat66d> H_geo_list;
        std::vector<fine::Vec6d>  g_geo_list;
        std::vector<Eigen::Vector2d> residual_geo_list;
        std::vector<float> weight_geo_list; //-- 每个几何残差的分布权重

        std::vector<fine::Mat66d> H_pho_list;
        std::vector<fine::Vec6d>  g_pho_list;
        std::vector<double> cost_pho_list;
        std::vector<double> weight_pho_list; //-- 每个光度残差的分布权重
        
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
                    Eigen::Vector3d p1_i(mvGeometryPoints[i].x_3d, 
                                        mvGeometryPoints[i].y_3d, 
                                        mvGeometryPoints[i].z_3d);
                    
                    // 跳过无效点
                    const auto& asso = mvAssociatedLines[i];
                    if(asso.first.x() == FLT_MAX) continue;
                    
                    // 计算雅可比矩阵
                    Eigen::Matrix<double, 6, 6> H_curr = Eigen::Matrix<double, 6, 6>::Zero();
                    Eigen::Matrix<double, 6, 1> g_curr = Eigen::Matrix<double, 6, 1>::Zero();
                    Eigen::Vector2d r_curr;
                    calculateJacobiPointLine3D2D(asso.first, asso.second, p1_i, H_curr, g_curr, r_curr);
                    
                    // 加权
                    double weight = mvGeometryPoints[i].score_depth;
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
                    // 这里可以使用锁或原子操作，但concurrent_vector更高效
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


        // //-- 更新光度残差的雅克比矩阵
        // 使用并行reduce计算总cost，同时安全地填充列表
        current_cost_pho = tbb::parallel_reduce(
            tbb::blocked_range<size_t>(0, N_1),  // 迭代范围
            0.0,                                // 初始值
            [&](const tbb::blocked_range<size_t>& r, double local_cost) {
                // 本地存储每个线程的计算结果
                std::vector<Eigen::Matrix<double, 6, 6>> local_H;
                std::vector<Eigen::Matrix<double, 6, 1>> local_g;
                std::vector<double> local_cost_list;
                
                for(size_t i = r.begin(); i != r.end(); ++i) {
                    // 获取2D点和深度
                    cv::Point pt_2d(mvGeometryPoints[i].x, mvGeometryPoints[i].y);
                    float depth = mvGeometryPoints[i].depth;
                    
                    // 计算雅可比矩阵
                    Eigen::Matrix<double, 6, 6> H_curr = Eigen::Matrix<double, 6, 6>::Zero();
                    Eigen::Matrix<double, 6, 1> g_curr = Eigen::Matrix<double, 6, 1>::Zero();
                    double cost_curr;
                    calculateJacobiPhotometric(img_cur_gray, img_ref_gray, pt_2d, depth,
                                            H_curr, g_curr, cost_curr);
                    
                    // 跳过无效点
                    if(g_curr.norm() == 0) continue;
                    
                    // 加权
                    double weight = mvGeometryPoints[i].score_depth;
                    double weight_sq = weight * weight;
                    
                    // 累加到本地存储
                    local_H.push_back(H_curr * weight_sq);
                    local_g.push_back(g_curr * weight_sq);
                    local_cost_list.push_back(cost_curr);
                    
                    // 累加本地cost
                    local_cost += cost_curr;
                }
                
                // 将本地结果合并到全局（加锁保护）
                {
                    static std::mutex mtx;
                    std::lock_guard<std::mutex> lock(mtx);
                    
                    H_pho_list.insert(H_pho_list.end(), local_H.begin(), local_H.end());
                    g_pho_list.insert(g_pho_list.end(), local_g.begin(), local_g.end());
                    cost_pho_list.insert(cost_pho_list.end(), local_cost_list.begin(), local_cost_list.end());
                }
                
                return local_cost;
            },
            // 合并各个线程的local_cost
            [](double a, double b) { return a + b; }
        );

        //-- 光度残差构建完之后，对光度残差进行卡方检验以进行加权
        eslam_core::robustWeightChi2 dst_pho(cost_pho_list); //-- 认为cost_geo_list满足卡方分布
        for(int i = 0; i < H_pho_list.size(); ++i){
            double weight_dist = dst_pho.computeWight(cost_pho_list[i]);
            weight_pho_list.push_back(weight_dist);
        }

        //-- 总的增量方程矩阵
        Eigen::Matrix<double , 6, 6> H = Eigen::Matrix<double , 6, 6>::Zero();
        Eigen::Matrix<double , 6, 1> g = Eigen::Matrix<double , 6, 1>::Zero();

        double geo_pho_ratio = geo_photo_ratio; //-- 几何残差与光度残差比例
        
        //-- 将所有残差块都更新到总的增量方程矩阵中
        for(int i = 0; i < H_geo_list.size(); ++i){
            H += H_geo_list[i] * weight_geo_list[i] * geo_pho_ratio;
            g += g_geo_list[i] * weight_geo_list[i] * geo_pho_ratio;
        }
        
        // for(int i = 0; i < H_pho_list.size(); ++i){
        //     H += H_pho_list[i] * weight_pho_list[i] / double(255 * 255 * 9);
        //     g += g_pho_list[i] * weight_pho_list[i] / double(255 * 255 * 9);
        // }

        // H /= double(H_geo_list.size() + H_pho_list.size());
        // g /= double(g_geo_list.size() + g_pho_list.size());

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
        T_cur_ref = Sophus::SE3d::exp(dx) * T_cur_ref;
        // last_cost = current_cost_geo + current_cost_pho ;
        last_cost = current_cost_geo;

        if(dx.norm() < 1e-6){
            break;
        }
    }
}

void FineTracker::estimate(Sophus::SE3d &T21, bool use_parallel)
{
    assert(mpF_ref != nullptr && mpF_cur != nullptr);

    associationRef2CurParallel();

    RegistrationCombinedParallel();

    //-- 此时得到 T_cur_ref

    //-- 求逆获得参考帧到当前帧的位姿变换
    T21 = T_cur_ref.inverse();
}