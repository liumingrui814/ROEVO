#include "edgeSelector.h"

float edgeSelector::calcAngleBias(float angle_1, float angle_2)
{
    float res = fabs(angle_1 - angle_2);
    if(res > 180){
        res = 360 - res;
    }
    return res;
}

cv::Mat edgeSelector::visualizeEdges()
{
    cv::RNG rng(66);
    cv::Mat image_viz(mMatCanny.rows, mMatCanny.cols, CV_8UC3, cv::Scalar::all(0));
    image_viz = mMatCanny.clone();
    if(image_viz.channels() == 3){
        cv::cvtColor(image_viz, image_viz, cv::COLOR_BGR2GRAY);
    }
    cv::cvtColor(image_viz, image_viz, cv::COLOR_GRAY2BGR);
    int maxLabel = 0;
    for(int i = 0; i < mvEdges.size(); ++i){
        int b = rng.uniform(0, 255);
        int g = rng.uniform(0, 255);
        int r = rng.uniform(0, 255);
        cv::Vec3b color = cv::Vec3b(b,g,r);
        //if(mvEdgeClusters[i].mvPoints.size()<15) continue;
        for(int j = 0; j < mvEdges[i].mvPoints.size(); ++j){
            orderedEdgePoint curr = mvEdges[i].mvPoints[j];
            image_viz.at<cv::Vec3b>(curr.y,curr.x) = color;
            cv::circle(image_viz, cv::Point(curr.x, curr.y), 1, color, 1, cv::LINE_AA);
        }
    }
    return image_viz;
}

cv::Mat edgeSelector::visualizeEdgesOrganized()
{
     cv::Mat valueTabel(256,1,CV_8UC1);
    cv::Mat ColorTabel;
    for(int i = 0; i<256; i++){
        valueTabel.at<uint8_t>(i,0)=i;
    }
    cv::applyColorMap(valueTabel,ColorTabel,cv::COLORMAP_PARULA);
    cv::Mat image_viz(mMatCanny.rows, mMatCanny.cols, CV_8UC3, cv::Scalar::all(0));
    image_viz = mMatCanny.clone();
    if(image_viz.channels() == 3){
        cv::cvtColor(image_viz, image_viz, cv::COLOR_BGR2GRAY);
    }
    cv::cvtColor(image_viz, image_viz, cv::COLOR_GRAY2BGR);
    int maxLabel = 0;
    for(int i = 0; i < mvEdges.size(); ++i){
        //if(mvEdgeClusters[i].mvPoints.size()<15) continue;
        for(int j = 0; j < mvEdges[i].mvPoints.size(); ++j){
            float proportion = float(j)/float(mvEdges[i].mvPoints.size());
            int idx = cvRound(proportion * 255);
            orderedEdgePoint curr = mvEdges[i].mvPoints[j];
            cv::Vec3b color = ColorTabel.at<cv::Vec3b>(idx,0);
            image_viz.at<cv::Vec3b>(curr.y,curr.x) = color;
            cv::circle(image_viz, cv::Point(curr.x, curr.y), 1, color, 1, cv::LINE_AA);
        }
    }
    return image_viz;
}

void edgeSelector::regionGrowthClusteringOCanny(float angle_Thres)
{
    cv::Mat labelMatTmp(mMatCanny.rows, mMatCanny.cols, CV_16UC1, cv::Scalar::all(65535));
    //-- 判断有没有遍历到当前点的矩阵 0 表示没有遍历到，1 表示遍历到
    cv::Mat visitedMat(mMatCanny.rows, mMatCanny.cols, CV_8UC1, cv::Scalar::all(0));
    //-- 清空edgemvEdgeClusters
    mvEdgeClusters.clear();
    
    //-- 预定义图像的指针
    uint8_t* canny_ptr = mMatCanny.data;
    uint16_t* label_ptr = (uint16_t*)labelMatTmp.data;
    uint8_t* visited_ptr = visitedMat.data;
    const float* angle_ptr = mMatGradAngle.ptr<float>(0);
    const int canny_step = mMatCanny.step;
    const int label_step = labelMatTmp.step / sizeof(uint16_t);;
    const int visited_step = visitedMat.step;
    const int angle_step = mMatGradAngle.step / sizeof(float);

    int label_global = 0;
    for(int y = 0; y < mMatCanny.rows; ++y)
    {
        for(int x = 0; x < mMatCanny.cols; ++x)
        {
            //-- 如果该点是canny得到的边缘且当前没遍历过这个点，就以该点为初始，区域生长地找到一个区域
            if (visited_ptr[y * visited_step + x] != 0 || canny_ptr[y * canny_step + x] != 255)
                continue;
            
            //-- 创建一个新的聚类
            std::vector<edgePoint> current_cluster;
            label_global++;
            //-- 更新当前位置的访问状态
            visited_ptr[y * visited_step + x] = 1;
            //-- 每个 cluster 中的点均会获得从0开始的编号
            int point_id = 0;

            //-- 创建一个边缘点，由于是该边缘的第一个点，设置ID为0，没有父节点，设为-1
            edgePoint curr_edge_point(cv::Point(x,y), point_id, -1);
            curr_edge_point.isRoot = true;
            point_id++;

            //-- 广度优先的区域生长方法的遍历队列
            std::queue<edgePoint> open_list;
            open_list.push(curr_edge_point);

            //-- 基于图搜索（深度优先或广度优先）的区域生长遍历
            while(!open_list.empty())
            {
                //-- 队列使用front()获得队首元素
                edgePoint current_point = open_list.front();
                open_list.pop();

                const int cx = current_point.pixel.x;
                const int cy = current_point.pixel.y;

                // if(mMatCanny.at<uint8_t>(current_point.pixel.y, current_point.pixel.x)==0) continue;

                //-- 能够进入open_list的像素必然属于当前一个聚类
                label_ptr[cy * label_step + cx] = label_global;
                current_cluster.push_back(current_point);

                //-- 扩展当前这个像素的相邻区域，得到新的满足条件的像素加入open_list
                const float curr_angle = angle_ptr[cy * angle_step + cx];

                //-- 定义一个 lambda 方法来扩展邻域, 扩展出的邻域需要满足
                //-- ① 坐标在图像区域内
                //-- ② 没有被访问过
                //-- ③ 是canny边缘点
                //-- ④ 梯度角度与它的父节点连续
                auto check_and_push = [&](int nx, int ny) {
                    if (nx >= 0 && nx < mWidth && ny >= 0 && ny < mHeight 
                        && visited_ptr[ny * visited_step + nx] == 0
                        && canny_ptr[ny * canny_step + nx] == 255) 
                    {
                        //-- 选择在图像区域内且未被访问的点，得到它的梯度角度
                        float neigh_angle = angle_ptr[ny * angle_step + nx];
                        if (calcAngleBias(neigh_angle, curr_angle) < angle_Thres) 
                        {
                            //-- 角度满足要求的点已经是可以聚类的点，因此可以赋ID，放到 queue 中继续扩展
                            visited_ptr[ny * visited_step + nx] = 1;
                            //-- 该点由 current_point 扩展得到，因此其父 ID 为 current_point.point_id
                            edgePoint neigh_pt(cv::Point(nx, ny), point_id, current_point.point_id);
                            open_list.push(neigh_pt);
                            point_id++;
                        }
                    }
                };
                //-- 向 8 邻域进行扩展
                check_and_push(cx+1, cy);   //-- 右
                check_and_push(cx+1, cy+1); //-- 右下
                check_and_push(cx,   cy+1); //-- 下
                check_and_push(cx-1, cy+1); //-- 左下
                check_and_push(cx-1, cy);   //-- 左
                check_and_push(cx-1, cy-1); //-- 左上
                check_and_push(cx,   cy-1); //-- 上
                check_and_push(cx+1, cy-1); //-- 右上

            }
            
            //-- region growth 结束，一组聚类生成，判断聚类大小，只取大序列
            if(current_cluster.size()>9)
            {
                //-- 结束一组聚类，此时得到一个完整的current_cluster
                EdgeCluster edge(current_cluster);
                mvEdgeClusters.push_back(edge);
            }
        }
    }
}

//-- 重新处理Canny的输出结果，对于四个方向的弯钩形状的结构需要将其中间的连接块剃掉
void edgeSelector::preprocessCannyMat()
{
    cv::Mat matBinary;
    mMatCanny.convertTo(matBinary, CV_8U, 1.0/255); // 直接转换为0/1值

    for(int i = 0; i < matBinary.rows; ++i) 
    {
        uint8_t* current = matBinary.ptr<uint8_t>(i);
        uint8_t* above = i > 0 ? matBinary.ptr<uint8_t>(i-1) : nullptr;
        uint8_t* below = i < matBinary.rows-1 ? matBinary.ptr<uint8_t>(i+1) : nullptr;
        
        for(int j = 0; j < matBinary.cols; ++j) 
        {
            if(current[j] == 0) continue; // 跳过非边缘点
            
            int left = j > 0 ? current[j-1] : 0;
            int right = j < matBinary.cols-1 ? current[j+1] : 0;
            int up = above ? above[j] : 0;
            int down = below ? below[j] : 0;
            
            bool connected = (left > 0 && up > 0) || (right > 0 && up > 0) ||
                             (left > 0 && down > 0) || (right > 0 && down > 0);
            
            if(connected) {
                current[j] = 0;
            }
        }
    }
}

void edgeSelector::processImage(const cv::Mat& image)
{

    mvEdgeClusters.clear();
    mvEdges.clear();

    if(image.channels() == 3)
    {
        cv::cvtColor(image, mMatGray, cv::COLOR_BGR2GRAY);
        mMatRGB = image.clone();
    }else if(image.channels()==1)
    {
        mMatGray = image.clone();
        cv::cvtColor(mMatGray, mMatRGB, cv::COLOR_GRAY2BGR);
    }
    mHeight = image.rows;
    mWidth = image.cols;

    cv::Mat grad_x, grad_y;
    cv::Scharr(mMatGray, grad_x, CV_32F, 1, 0);
    cv::Scharr(mMatGray, grad_y, CV_32F, 0, 1);

    // 预分配内存
    mMatGradMagnitude.create(image.size(), CV_32F);
    mMatGradAngle.create(image.size(), CV_32F);

    //-- 计算梯度幅值和方向, magnitude是大小，angle是方向，取值是0~360度
    //-- 最后一个值是false就是L1范数的梯度模值，true就是L2范数的梯度模值
    cv::cartToPolar(grad_x, grad_y, mMatGradMagnitude, mMatGradAngle, true);

    if(mbUseFixedThreshold)
    {
        cv::Canny(image, mMatCanny, mpCanny_lower_bound, mpCanny_higher_bound, 3, true); 
    }else{
        cv::Mat binary;
        double otsu_thresh = cv::threshold(image, binary, 0, 255, cv::THRESH_OTSU);
        cv::Canny(image, mMatCanny, 0.5*otsu_thresh, otsu_thresh, 3, true);
    }
    preprocessCannyMat();
    regionGrowthClusteringOCanny(mpAngle_bias);
    // cvt2OrderedEdges();
    cvt2OrderedEdgesParallel();
}

void edgeSelector::cvt2OrderedEdges()
{
    // 预分配 mvEdges 内存（减少动态扩容开销）
    mvEdges.reserve(mvEdgeClusters.size());

    // 提前获取 mMatGradAngle 的指针（避免重复调用 at<>）
    const float* angle_ptr = mMatGradAngle.ptr<float>(0);
    const int angle_step = mMatGradAngle.step / sizeof(float);

    for (int i = 0; i < mvEdgeClusters.size(); ++i) {
        Edge curr_edge(i);
        const auto& cluster_points = mvEdgeClusters[i].organize();;
        curr_edge.mvPoints.reserve(cluster_points.size());

        for (const auto& point : cluster_points) {  // 范围 for 循环更高效
            const int x = static_cast<int>(point.pixel.x);
            const int y = static_cast<int>(point.pixel.y);

            // angle(0~360) of the image gradient
            const float angle = angle_ptr[y * angle_step + x];
            
            // an orderedEdgePoint is initially constructed by coordinate (x,y) and gradient angle
            curr_edge.mvPoints.emplace_back(x, y, angle);
        }
        mvEdges.push_back(std::move(curr_edge));  // 移动语义减少拷贝
    }
}

void edgeSelector::cvt2OrderedEdgesParallel()
{
    // 并行写入，因此直接resize而不是reverse
    mvEdges.resize(mvEdgeClusters.size());

    // 提前获取 mMatGradAngle 的指针（避免重复调用 at<>）
    const float* angle_ptr = mMatGradAngle.ptr<float>(0);
    const int angle_step = mMatGradAngle.step / sizeof(float);

    // 使用TBB并行处理每个边缘簇
    tbb::parallel_for(tbb::blocked_range<size_t>(0, mvEdgeClusters.size()),
        [&](const tbb::blocked_range<size_t>& range) {
            for (size_t i = range.begin(); i != range.end(); ++i) {
                Edge curr_edge(i);  // 每个边缘有自己独立的ID
                const auto& cluster_points = mvEdgeClusters[i].organize();
                curr_edge.mvPoints.reserve(cluster_points.size());

                for (const auto& point : cluster_points) {
                    const int x = static_cast<int>(point.pixel.x);
                    const int y = static_cast<int>(point.pixel.y);

                    // angle(0~360) of the image gradient
                    const float angle = angle_ptr[y * angle_step + x];
                    
                    // an orderedEdgePoint is initially constructed by coordinate (x,y) and gradient angle
                    curr_edge.mvPoints.emplace_back(x, y, angle);
                }
                mvEdges[i] = std::move(curr_edge);  // 直接写入到预分配的位置
            }
        });
}