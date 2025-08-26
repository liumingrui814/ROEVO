#ifndef FEATURE_MERGER_H
#define FEATURE_MERGER_H

#include "KeyFrame.h"
#include "disjointSet.h"
#include "elementEdge.h"

namespace edge_map{

class featureMerger{
public:

    // 禁止实例化
    featureMerger() = delete;

    enum MergeType {
        CANNOT_MERGE,
        APPEND_AS_IS,
        PREPEND_AS_IS,
        APPEND_REVERSED,
        PREPEND_REVERSED
    };

    // 
    static constexpr int MAX_VALUE = 100;

    //-- 参数分别为
    //-- 1. cluster中的所有elementEdge, 
    //-- 2. 每个elementEdge的关键帧索引，
    //-- 3. 关键帧列表，4. 返回的融合边缘
    //-- 5. 构成融合边缘的所有有效elementEdge
    static void getMergedClusterProjection(const std::vector<elementEdge>& vElements, 
                        const std::vector<int>& vKFindices,
                        const std::vector<KeyFramePtr>& vKeyFrames,
                        std::vector<cv::Point3d>& mergedCloud_global,
                        std::vector<int>& involved_elements);
    
    //-- 通过周围边缘向基准边缘投影实现增量构建与边缘fitting
    //-- 1. cluster中的所有elementEdge, 
    //-- 2. 每个elementEdge的关键帧索引，
    //-- 3. 关键帧列表，4. 返回的融合边缘
    //-- 5. 构成融合边缘的所有有效elementEdge
    static void getMergedClusterIncremental(const std::vector<elementEdge>& vElements, 
        const std::vector<int>& vKFindices,
        const std::vector<KeyFramePtr>& vKeyFrames,
        std::vector<cv::Point3d>& mergedCloud_global,
        std::vector<int>& involved_elements);

    static void getMergedClusterIterative(const std::vector<elementEdge>& vElements, 
        const std::vector<int>& vKFindices,
        const std::vector<KeyFramePtr>& vKeyFrames,
        std::vector<cv::Point3d>& mergedCloud_global,
        double& avgDistThres,
        std::vector<int>& involved_elements);

private:
    static double statisticFilter(std::vector<Eigen::Vector3d>& point_cluster, std::vector<float>& scores, double std_mult = 1.0);

    static std::tuple<cv::Point2d, bool> calculateFootAndCheck(cv::Point2d A, cv::Point2d B, cv::Point2d C);

    static std::vector<size_t> findLongestFalseSegment(const std::vector<bool>& asso_list);

    static void assoSurround2Ref(std::vector<Edge>& vEdges,
        const std::vector<Sophus::SE3d>& vKFposes,
        size_t ref_edge_index, Sophus::SE3d pose_ref,
        float fx, float fy, float cx, float cy,
        const std::vector<cv::Point2d>& ref_edge_sampled_2d,
        std::vector<std::vector<Eigen::Vector3d>>& point_neighbors_total,
        std::vector<std::vector<float>>& point_neighbors_score_total,
        std::vector<std::vector<bool>>& associatedList_total,
        std::vector<int>& statistics);
    
    static void sampleOEdgeUniform(std::vector<cv::Point2d>& sampled_longest_edge,
        std::vector<Eigen::Vector3d>& sampled_longest_edge_3d,
        const std::vector<orderedEdgePoint>& edgePoints, const int sampleBias);

    static bool isMergable(const std::vector<Eigen::Vector3d>& ref_seq,
            const std::vector<Eigen::Vector3d>& add_seq,
            MergeType& merge_type,
            double distance_threshold = 0.01);

    //-- 模板拼接函数（直接在switch中实现所有逻辑）
    template<typename T>
    static void mergeVectors(std::vector<T>& ref_seq,
                            const std::vector<T>& add_seq,
                            MergeType merge_type) {
        switch (merge_type) {
            case APPEND_AS_IS:
                // 正向追加到末尾
                ref_seq.insert(ref_seq.end(), add_seq.begin(), add_seq.end());
                break;
                
            case PREPEND_AS_IS:
                // 正向插入到开头
                ref_seq.insert(ref_seq.begin(), add_seq.begin(), add_seq.end());
                break;
                
            case APPEND_REVERSED:
                // 反向追加到末尾
                ref_seq.insert(ref_seq.end(), add_seq.rbegin(), add_seq.rend());
                break;
                
            case PREPEND_REVERSED:
                // 反向插入到开头
                ref_seq.insert(ref_seq.begin(), add_seq.rbegin(), add_seq.rend());
                break;
                
            case CANNOT_MERGE:
            default:
                // 不执行任何操作
                break;
        }
    }
};

}

#endif