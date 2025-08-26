#include "localMap.h"
using namespace edge_map;

//-- 在关键帧数量大于2的时候初始化一个局部地图
void localMap::initLocalMap()
{
    assert(mvKeyFrames.size() == 2);

    //-- 获取三个关键帧
    KeyFramePtr pKF_0 = mvKeyFrames[0];
    KeyFramePtr pKF_1 = mvKeyFrames[1];

    //* STEP-0 首先将 pKF_0 完整加入地图
    for(int i = 0; i < pKF_0->mvEdges.size(); ++i)
    {
        //-- 关键帧中的边缘都是有效边缘
        int edge_idx = i;
        elementEdge ele(pKF_0->KF_ID, edge_idx);
        //-- 局部地图添加当前的边缘元素
        mvElementEdges.push_back(ele);
        mmElementID2index[ele.element_id] = mvElementEdges.size()-1;
        //-- 关键帧索引当前的局部地图
        pKF_0->mmEdgeIndex2ElementEdgeID[i] = ele.element_id;
    }

    //-- 初始化整个局部地图的基准坐标系
    // T_ref = pKF_0->KF_pose_g;

    //-- 两帧关联
    associationResult res = associationMulti2Multi(pKF_0, pKF_1);

    //*  归纳关联结果
    //-- 构建索引, first是边缘类别的id, second 是边缘类别对应的 pKF_0 边缘的 root_idx
    std::unordered_map<int, int> labelRootMap;
    std::unordered_map<int, int> edgeLabelMap = std::move(res.second); //-- 边缘-label map
    std::unordered_map<int, int> curRefMap = std::move(res.first);     //-- 当前帧-参考帧 map

    //-- 标记 pKF_0 中哪些边缘可关联哪些不可
    std::vector<bool> isAssociated(pKF_0->mvEdges.size(), false);

    // * STEP-1 pKF_0 内部进行并查集合并
    for(const auto &pair : edgeLabelMap)
    {
        const int label = pair.second;
        const int edge_idx = pair.first;

        //-- 能遍历到说明可关联
        isAssociated[edge_idx] = true;

        if(labelRootMap.find(label) == labelRootMap.end())
        {
            labelRootMap[label] = edge_idx;
        }else{
            //-- 得到当前index的 root_idx
            int root_idx = labelRootMap[label];

            //-- 由帧边缘ID对应到地图的边缘ID，并使用并查集进行合并这些边缘
            unsigned int map_id_1 = pKF_0->mmEdgeIndex2ElementEdgeID.at(root_idx);
            unsigned int map_id_2 = pKF_0->mmEdgeIndex2ElementEdgeID.at(edge_idx);

            //-- 并查集的归并
            int map_idx_2 = mmElementID2index.at(map_id_2);
            mvElementEdges[map_idx_2].union_id = findRoot(map_id_1);
        }
    }

    // * STEP-2 遍历 pKF_1 的关联列表，将能够关联的边缘加入Map中
    for(const auto& pair: curRefMap)
    {
        int cur_idx = pair.first;
        int ref_idx = pair.second;

        int label_id = edgeLabelMap[ref_idx];

        //-- 如果被关联到的参考帧是无效帧，在frame2MapElement中找不到，则跳过
        if(pKF_0->mmEdgeIndex2ElementEdgeID.find(ref_idx)==pKF_0->mmEdgeIndex2ElementEdgeID.end()) continue;

        int size = mvElementEdges.size();
        elementEdge ele(pKF_1->KF_ID, cur_idx);
        //-- 与当前帧的该边缘关联的参考帧的地图id
        ele.union_id = pKF_0->mmEdgeIndex2ElementEdgeID[ref_idx];

        mvElementEdges.push_back(ele);
        mmElementID2index[ele.element_id] = mvElementEdges.size()-1;
        pKF_1->mmEdgeIndex2ElementEdgeID[cur_idx] = ele.element_id;
    }

    // * STEP-3 将 pKF_0 中不能与 pKF_1 关联的边缘移除出局部地图
    for (auto it = mvElementEdges.begin(); it != mvElementEdges.end(); ) {
        if(it->kf_id == pKF_0->KF_ID && !isAssociated[it->kf_edge_idx]) 
        {
            //-- 首先删除关键帧到局部地图的索引
            if (pKF_0->mmEdgeIndex2ElementEdgeID.find(it->kf_edge_idx) != pKF_0->mmEdgeIndex2ElementEdgeID.end())
            {
                pKF_0->mmEdgeIndex2ElementEdgeID.erase(it->kf_edge_idx);
            }
            //-- 再删除局部地图的 elementEdge
            it = mvElementEdges.erase(it); // erase 返回下一个有效迭代器
        }else{
            ++it; // 否则继续
        }
    }
    //-- 有删除元素即需要重置 id2idx 的map
    mmElementID2index.clear();
    for(size_t i = 0; i < mvElementEdges.size(); ++i)
    {
        elementEdge& ele = mvElementEdges[i];
        mmElementID2index[ele.element_id] = i;
    }

    //-- 剪枝
    pruningMap();

    //-- 生成 elementEdge Cluster
    //-- 边缘地图元素与类别ID的映射, 
    //-- first是map element的label，second是map element的列表，即一团map element是一个完整边缘
    std::map<unsigned int, std::vector<unsigned int>> eleEdgeMap;
    eleEdgeMap.clear();
    for(int i = 0; i < mvElementEdges.size(); ++i)
    {
        unsigned int curr_id = mvElementEdges[i].element_id;
        unsigned int root_id = findRoot(curr_id);
        //std::cout<<root_idx<<std::endl;
        //-- root_idx是0的时候会有BUG, 需要检查
        eleEdgeMap[root_id].push_back(mvElementEdges[i].element_id);
    }

    for (auto& pair : eleEdgeMap)
    {
        unsigned int cluster_id = pair.first;
        std::vector<unsigned int> clusters = pair.second;
        elementEdgeCluster e_cluster(cluster_id, clusters);
        mvEleEdgeClusters.push_back(std::move(e_cluster));
        mmClusterID2index[cluster_id] = mvEleEdgeClusters.size() - 1;
    }

    mmKFID2KFindex[pKF_0->KF_ID] = 0;
    mmKFID2KFindex[pKF_1->KF_ID] = 1;

}

//-- 输入 curr_id 是当前 elementEdge 的 element_id
unsigned int localMap::findRoot(unsigned int curr_id)
{
    //-- 与当前帧同类的根节点
    int curr_idx = mmElementID2index.at(curr_id);
    int union_id = mvElementEdges.at(curr_idx).union_id;
    if(union_id == -1){//-- 如果当前的curr_id的根节点就是-1，说明这个就是独立的根节点
        return curr_id;
    }else{//-- 如果不是，就递归找根节点
        return findRoot(union_id);
    }
}

void localMap::pruningMap(){
    for(int i = 0; i < mvElementEdges.size(); ++i){
        unsigned int ele_id = mvElementEdges[i].element_id;
        unsigned int root_id = findRoot(ele_id);
        //std::cout<<root_id << "," << mvElementEdges[i].union_id <<std::endl;
        if (root_id != ele_id)
            mvElementEdges[i].union_id = root_id;
    }
}

//-- 将 frame_cur 投影到 frame_ref 上实现 frame_cur 与 frame_ref 的多对多关联
//-- 默认 frame_ref 时间戳在 frame_cur 之前
associationResult localMap::associationMulti2Multi(KeyFramePtr frame_ref, KeyFramePtr frame_cur)
{
    //-- 为参考帧的边缘创建一个并查集，并在后续持续的优化过程中更新这个并查集
    const int ref_edge_num = frame_ref->mvEdges.size();
    DisjointSet edgeRefSet(ref_edge_num);

    std::vector<bool> isAssociated(ref_edge_num, false); //-- 有的参考帧的边并没有被关联上，这里记录一下

    //-- 由于是当前帧重投影到参考帧关联参考帧，故而是当前帧存储与参考帧的关联关系
    std::unordered_map<int, int> currframeMap; //-- <idx1, idx2> 记录当前帧的第idx1条边与参考帧的第idx2条边是一类的
    currframeMap.reserve(frame_cur->mvEdges.size() / 2);

    //-- 位姿差
    Sophus::SE3d T_ref_cur = frame_ref->KF_pose_g.inverse() * frame_cur->KF_pose_g;

    //-- 缓存参考帧的mmIndexMap以避免重复查找
    const auto& refIndexMap = frame_ref->mmIndexMap;

    //-- 用当前帧的有效边缘去参考帧中关联参考帧的边缘，并且对关联到多个的参考帧关联进行并查集的合并
    const int cur_edge_num = frame_cur->mvEdges.size();
    for(int i = 0; i < cur_edge_num; ++i)
    {
        Edge& query_edge = frame_cur->mvEdges[i];
        //-- 让当前帧的一条边缘关联参考帧
        std::vector<int> associated_edges = frame_ref->edgeWiseCorrespondenceLocalMapping(query_edge,T_ref_cur);
        
        if(associated_edges.empty()) continue;
        
        //-- 此时整个返回的列表里的边缘都需要进行合并
        if(associated_edges.size() >= 2)
        {
            //-- associated_edges[0]是第0条边的ID
            const int root_idx = refIndexMap.at(associated_edges[0]);

            for(size_t j = 1; j < associated_edges.size(); j++)
            {
                //-- associated_edges[j]是这些被关联的边中的第j条边的ID
                int curr_idx = refIndexMap.at(associated_edges[j]);
                //-- 在并查集中合并这两个idx，即mvEdges[root_idx]与mvEdges[curr_idx]是相同的一个类
                edgeRefSet.to_union(root_idx, curr_idx);
            }
        }

        //-- 更新关联关系, 如果有建立正确的关联，则得到当前边与参考帧的关联
        currframeMap[i] = refIndexMap.at(associated_edges[0]);
        //-- 记录一下参考帧的这条边有没有被关联
        for(const auto& edge_id : associated_edges)
        {
            const int curr_idx = refIndexMap.at(edge_id);
            isAssociated[curr_idx] = true;
        }
    }
    //-- 至此多对多的聚类完成
    edgeRefSet.pruningSet();

    //-- 整理分类与可视化，需要为并查集中所有的类创建一个唯一的颜色，并在可视化时查找这个颜色并渲染
    std::unordered_map<int, int> clusterMap; //-- 分类与颜色对应的map，分类就是ref帧的并查集的根节点的idx
                                             //-- first是ref帧的每条边缘的idx，second是对应的类别
    clusterMap.reserve(ref_edge_num / 2);
    
    for(int i = 0; i < ref_edge_num; ++i){
        if(isAssociated[i]==false) continue;
        //-- 对于可关联的边，通过并查集找到其根节点
        clusterMap[i] = edgeRefSet.find(i);
    }
    associationResult res;
    res.first = std::move(currframeMap);
    res.second = std::move(clusterMap);
    //-- 预存储当前帧与参考帧的关联关系
    frame_cur->mmMapAssociations[frame_ref->KF_ID] = res;
    return res;
}

void localMap::addFrame2LocalMap(KeyFramePtr frame_cur)
{
    //-- 先确认有没有初始化好, 没有初始化好则后面全部不予执行
    if(msState == State::NOT_INITIALIZED)
    {
        if(mvKeyFrames.size() < 2)
        {
            mvKeyFrames.push_back(frame_cur);
            mmKFID2KFindex[frame_cur->KF_ID] = mvKeyFrames.size()-1;
        }
            
        if (mvKeyFrames.size() ==2)
        {
            initLocalMap();
            std::cout<<"finish init"<<std::endl;
            msState = State::INITIALIZED;
        }
        return;
    }

    //-- 先初始化所有的cluster的mbModifiedCur，以标记当前这些cluster有没有被修改
    for (auto& cluster : mvEleEdgeClusters)
    {
        cluster.mbModifiedCur = false;
    }

    int N = frame_cur->mvEdges.size();
    //-- 当前帧的关联列表，能与先前任一参考帧关联的都会被设为true
    std::vector<bool> isAssociated(N, false);
    frame_cur->mmEdgeIndex2ElementEdgeID.clear();

    for(int i = mvKeyFrames.size()-1; i >= 0; --i)
    {
        KeyFramePtr frame_ref = mvKeyFrames[i];
        associationResult res;
        //-- 先检查当前帧有没有和参考帧建立过关联
        if(frame_cur->mmMapAssociations.find(frame_ref->KF_ID) != frame_cur->mmMapAssociations.end()){
            //-- 关联过就不用重新执行关联了
            res = frame_cur->mmMapAssociations[frame_ref->KF_ID];
        }else{
            //-- 没关联过就要重新关联一下
            res = associationMulti2Multi(frame_ref, frame_cur);
        }

        //-- 归纳关联结果
        //-- 构建索引, first是边缘类别的id, second是边缘类别对应的ref边缘的idx
        std::unordered_map<int, int> labelRootMap;
        std::unordered_map<int, int> edgeLabelMap = std::move(res.second); //-- 边缘-label map
        std::unordered_map<int, int> curRefMap = std::move(res.first);     //-- 当前帧-参考帧 map

        //-- 参考帧内部进行并查集合并（参考帧自己的边缘是同一个边缘）
        for(const auto &pair : edgeLabelMap)
        {
            const int label = pair.second;
            const int edge_idx = pair.first; //-- 该edge_idx一定存在于local map中

            if(labelRootMap.find(label) == labelRootMap.end())
            {
                labelRootMap[label] = edge_idx;
            }else{
                int root_idx = labelRootMap[label];

                //-- 由帧边缘ID对应到地图的边缘ID，并使用并查集进行合并这些边缘
                unsigned int map_id_1 = frame_ref->mmEdgeIndex2ElementEdgeID.at(root_idx);
                unsigned int map_id_2 = frame_ref->mmEdgeIndex2ElementEdgeID.at(edge_idx);

                // * 并查集的归并
                int map_idx_2 = mmElementID2index.at(map_id_2);
                int map_idx_1 = mmElementID2index.at(map_id_1);
                //-- 检查需要归并的两个 elementEdge 是不是原本同类
                //-- 原本同类不需要额外处理，非同类需要归并
                unsigned int cluster_id_1 = findRoot(map_id_1);
                unsigned int cluster_id_2 = findRoot(map_id_2);
                if(cluster_id_1 != cluster_id_2)
                {
                    //-- 不是同类，需要merge两个cluster
                    int cluster_idx_1 = mmClusterID2index.at(cluster_id_1);
                    int cluster_idx_2 = mmClusterID2index.at(cluster_id_2);
                    
                    mergeElementCluster(cluster_idx_1, cluster_idx_2);

                }
            }
        }

        //-- 遍历当前帧的关联列表，将能够关联的帧加入Map中
        for(const auto& pair: curRefMap){
            int cur_idx = pair.first;
            //-- 如果当前边缘已经被其他参考帧关联了，那跳过这个边缘
            if(isAssociated[cur_idx]==true) continue;
            int ref_id = pair.second;
            int label_id = edgeLabelMap[ref_id];

            //-- 如果被关联到的参考帧是无效帧，在frame2MapElement中找不到，则跳过
            if(frame_ref->mmEdgeIndex2ElementEdgeID.find(ref_id)==frame_ref->mmEdgeIndex2ElementEdgeID.end()) continue;

            //-- 对于已经关联了的边缘，创建 elementEdge 并union
            isAssociated[cur_idx] = true;
            
            elementEdge ele(frame_cur->KF_ID, cur_idx);
            int map_id_1 = frame_ref->mmEdgeIndex2ElementEdgeID.at(ref_id);
            
            ele.union_id = findRoot(map_id_1);
            
            mvElementEdges.push_back(ele);
            mmElementID2index[ele.element_id] = mvElementEdges.size()-1;
            frame_cur->mmEdgeIndex2ElementEdgeID[cur_idx] = ele.element_id;

            //-- 将当前 elementEdge 归到 elementEdgeCluster 中
            unsigned int cluster_id = ele.union_id;
            if(mmClusterID2index.find(cluster_id)==mmClusterID2index.end())
            {
                std::cout<<"\033[31m [LOCAL MAPPING] \033[0m " << "can't locate root cluster" << std::endl;
            }
            int cluster_idx = mmClusterID2index[cluster_id];
            mvEleEdgeClusters[cluster_idx].mvElementEdgeIDs.push_back(ele.element_id);
            mvEleEdgeClusters[cluster_idx].mbModifiedCur = true;

        }
    }

    //-- 遍历当前帧未被关联的边缘，将其加入到地图中
    for(int i = 0; i < frame_cur->mvEdges.size(); ++i){
        if(isAssociated[i]== true) continue;
        //-- 对于未被关联的边缘，加入地图中
        int size = mvElementEdges.size();
        
        elementEdge ele(frame_cur->KF_ID, i);
        mvElementEdges.push_back(ele);
        mmElementID2index[ele.element_id] = mvElementEdges.size()-1;
        frame_cur->mmEdgeIndex2ElementEdgeID[i] = ele.element_id;

        unsigned int cluster_id = ele.element_id;
        std::vector<unsigned int> clusters;
        clusters.push_back(ele.element_id);
        elementEdgeCluster e_cluster(cluster_id, clusters);
        mvEleEdgeClusters.push_back(std::move(e_cluster));
        mmClusterID2index[cluster_id] = mvEleEdgeClusters.size() - 1;
        
    }
    //-- 将当前帧添加入Map
    mvKeyFrames.push_back(frame_cur);
    mmKFID2KFindex[frame_cur->KF_ID] = mvKeyFrames.size()-1;

    //-- 统计被修改的cluster 更新所有 cluster 的 count_not_update
    for (auto& cluster : mvEleEdgeClusters)
    {
        if(cluster.mbModifiedCur == false)
        {
            cluster.count_not_update += 1;
        }else{
            //-- 如果被修改就从0开始继续数
            cluster.count_not_update = 0;
        }
    }
    elementClusterCulling();
}

void localMap::mergeElementCluster(int cluster_idx_1, int cluster_idx_2)
{
    //-- 检查两个cluster里谁大小更大
    elementEdgeCluster& cluster_1 = mvEleEdgeClusters[cluster_idx_1];
    elementEdgeCluster& cluster_2 = mvEleEdgeClusters[cluster_idx_2];
    int size_1 = cluster_1.mvElementEdgeIDs.size();
    int size_2 = cluster_2.mvElementEdgeIDs.size();

    if(size_1 > size_2)
    {
        //-- size_1 更大就将 cluster_2 合并到 cluster_1;
        // * STEP-1 调整 cluster_2 中的所有边的根节点
        for(auto& id : cluster_2.mvElementEdgeIDs)
        {
            int idx = mmElementID2index[id];
            mvElementEdges[idx].union_id = cluster_1.cluster_id; //-- cluster_id 就是根节点id
        }
        // * STEP-2 将 cluster_2 中所有边移动到 cluster_1
        auto& vElementIDS_1 = cluster_1.mvElementEdgeIDs;
        vElementIDS_1.insert(vElementIDS_1.end(), cluster_2.mvElementEdgeIDs.begin(), cluster_2.mvElementEdgeIDs.end());
        
        // * STEP-3 删除 cluster_2
        mvEleEdgeClusters.erase(mvEleEdgeClusters.begin() + cluster_idx_2); // 删除第i个元素
        mmClusterID2index.clear();
        for (int i = 0; i < mvEleEdgeClusters.size(); ++i)
        {
            unsigned int c_id = mvEleEdgeClusters[i].cluster_id;
            mmClusterID2index[c_id] = i;
        }
        // * STEP-4 最新操作，因此记录 cluster_1 本轮被修改
        cluster_1.mbModifiedCur = true;

    }else{
        //-- size_2 更大就将 cluster_1 合并到 cluster_2;
        // * STEP-1 调整 cluster_2 中的所有边的根节点
        for(auto& id : cluster_1.mvElementEdgeIDs)
        {
            int idx = mmElementID2index[id];
            mvElementEdges[idx].union_id = cluster_2.cluster_id; //-- cluster_id 就是根节点id
        }
        // * STEP-2 将 cluster_1 中所有边移动到 cluster_2
        auto& vElementIDS_2 = cluster_2.mvElementEdgeIDs;
        vElementIDS_2.insert(vElementIDS_2.end(), cluster_1.mvElementEdgeIDs.begin(), cluster_1.mvElementEdgeIDs.end());
        
        // * STEP-3 删除 cluster_1
        mvEleEdgeClusters.erase(mvEleEdgeClusters.begin() + cluster_idx_1); // 删除第i个元素
        mmClusterID2index.clear();
        for (int i = 0; i < mvEleEdgeClusters.size(); ++i)
        {
            unsigned int c_id = mvEleEdgeClusters[i].cluster_id;
            mmClusterID2index[c_id] = i;
        }
        // * STEP-4 最新操作，因此记录 cluster_2 本轮被修改
        cluster_2.mbModifiedCur = true;
    }
}

void localMap::elementClusterCulling()
{
    std::vector<int> ele_index_tobe_deleted;

    for(auto cluster_iter = mvEleEdgeClusters.begin(); cluster_iter != mvEleEdgeClusters.end(); )
    {
        if(cluster_iter->count_not_update > 3)
        {
            //-- 删除
            // * STEP-1 先删除该cluster中的所有边缘
            int N = cluster_iter->mvElementEdgeIDs.size();
            std::vector<int> indicesToDelete;
            indicesToDelete.reserve(N);
            for(int i = 0; i < N; ++i)
            {
                //-- 删除 ele_id 对应的 elementEdge
                unsigned int ele_id = cluster_iter->mvElementEdgeIDs[i];
                int ele_idx = mmElementID2index[ele_id];
                elementEdge& ele_edge = mvElementEdges[ele_idx];

                //-- substep-1 删除关键帧对该 elementEdge 的索引
                KeyFramePtr pKF = mvKeyFrames.at(mmKFID2KFindex[ele_edge.kf_id]);
                pKF->mmEdgeIndex2ElementEdgeID.erase(ele_edge.kf_edge_idx);
                
                //-- substep-2 记录要删除的 elementEdge 的索引
                indicesToDelete.push_back(ele_idx);
            }
            //-- substep-3 整理记录的要删除的 elementEdge
            ele_index_tobe_deleted.insert(ele_index_tobe_deleted.end(), indicesToDelete.begin(), indicesToDelete.end());

            // * STEP-2 删除该 cluster
            cluster_iter = mvEleEdgeClusters.erase(cluster_iter);
        }else{
            cluster_iter++;
        }
    }

    // * STEP-3 根据整理出来的要删除的 elementEdge 做统一删除
    std::sort(ele_index_tobe_deleted.begin(), ele_index_tobe_deleted.end(), std::greater<int>());
    for(int idx : ele_index_tobe_deleted) {
        mvElementEdges.erase(mvElementEdges.begin() + idx);
    }
    //-- 重置 elementEdge 的索引映射
    mmElementID2index.clear();
    for(size_t i = 0; i < mvElementEdges.size(); ++i)
    {
        elementEdge& ele = mvElementEdges[i];
        mmElementID2index[ele.element_id] = i;
    }

    // * STEP-4 重新整理 cluster 的 ID-索引 映射
    mmClusterID2index.clear();
    for (int i = 0; i < mvEleEdgeClusters.size(); ++i)
    {
        unsigned int c_id = mvEleEdgeClusters[i].cluster_id;
        mmClusterID2index[c_id] = i;
    }

}

void localMap::removeKeyFrameFront()
{
    assert(mvKeyFrames.size() > 0);

    KeyFramePtr pKF_del = mvKeyFrames[0];
    int kf_id = pKF_del->KF_ID;

    std::cout<<"remove frame "<<kf_id<<std::endl;

    // * STEP-1 找到 elementEdge 中所有来自该关键帧的边缘
    std::vector<int> indicesToDelete;
    indicesToDelete.reserve(pKF_del->mvEdges.size());

    std::set<unsigned int> clusters_tobe_culled;

    for(size_t i = 0; i < mvElementEdges.size(); ++i)
    {
        elementEdge& ele_curr = mvElementEdges[i];
        if(ele_curr.kf_id == kf_id)
        {
            indicesToDelete.push_back(i);
            //-- 找这个 elementEdge 的 cluster
            unsigned int cluster_id = findRoot(ele_curr.element_id);
            if(mmClusterID2index.find(cluster_id) != mmClusterID2index.end())
            {
                //-- 标记这个 cluster，其需要后续被清理
                clusters_tobe_culled.insert(cluster_id);
            }else{
                std::cout<<cluster_id<<std::endl;
                std::cout<<"\033[31m [ERROR] \033[0m" << "locate a cluster_id that doesn't match root_id"<<std::endl;
            }
        }
    }

    // * STEP-2 从 cluster 中删去这些边缘的索引
    std::vector<int> clusterIdxToDelete;
    clusterIdxToDelete.reserve(mvEleEdgeClusters.size()/10);
    for(auto id = clusters_tobe_culled.begin(); id != clusters_tobe_culled.end(); ++id)
    {
        int index = mmClusterID2index[*id];
        elementEdgeCluster& cluster = mvEleEdgeClusters[index];
        std::vector<int> indicesToDelete_ele;
        int N = cluster.mvElementEdgeIDs.size();
        indicesToDelete_ele.reserve(N/2);
        
        for(int i = 0; i < N; ++i)
        {
            unsigned int ele_id = cluster.mvElementEdgeIDs[i];
            int ele_idx = mmElementID2index[ele_id];
            if(mvElementEdges[ele_idx].kf_id == kf_id)
            {
                //-- 标记这个位置
                indicesToDelete_ele.push_back(i);
            }
        }

        //-- 删除这个cluster中的该关键帧的边缘索引
        std::sort(indicesToDelete_ele.begin(), indicesToDelete_ele.end(), std::greater<int>());
        for(int idx : indicesToDelete_ele) {
            cluster.mvElementEdgeIDs.erase(cluster.mvElementEdgeIDs.begin() + idx);
        }

        //-- 如果这个 cluster 空了，那删除这个cluster, 此处做标记
        if(cluster.mvElementEdgeIDs.size()==0)
        {
            clusterIdxToDelete.push_back(index);
            continue;
        }

        //-- 如果这个 cluster 非空，删除之后该 cluster_id 需要重新赋值，以防旧的 root 被删除
        //-- 让这个cluster 中的所有 elementEdge 将其中的任意边缘作为 root
        int new_N = cluster.mvElementEdgeIDs.size();
        unsigned int ele_id_new_root = cluster.mvElementEdgeIDs[0];
        int ele_idx_new_root = mmElementID2index[ele_id_new_root];
        mvElementEdges[ele_idx_new_root].union_id = -1;
        for(size_t i = 1; i < new_N; ++i)
        {
            unsigned int ele_id = cluster.mvElementEdgeIDs[i];
            int ele_idx = mmElementID2index[ele_id];
            mvElementEdges[ele_idx].union_id = mvElementEdges[ele_idx_new_root].element_id;
        }
        //-- 重置该 cluster 的 cluster_id
        if(mmClusterID2index.find(ele_id_new_root) == mmClusterID2index.end())
        {
            cluster.cluster_id = ele_id_new_root;
        }else{
            if(cluster.cluster_id != ele_id_new_root){
                //-- 理论上不可能有其他的 cluster 的 cluster_id 是这个 cluster 中的elementEdge 的 id 
                std::cout<<"\033[31m [ERROR] \033[0m"<<"wrong cluster construction"<<std::endl;
            }
        }
    }

    //-- 删除 cluster
    std::sort(clusterIdxToDelete.begin(), clusterIdxToDelete.end(), std::greater<int>());
    for(int idx : clusterIdxToDelete) {
        mvEleEdgeClusters.erase(mvEleEdgeClusters.begin() + idx);
    }
    //-- 重新整理 cluster 的 id-index 映射
    mmClusterID2index.clear();
    for(int i = 0; i < mvEleEdgeClusters.size(); ++i)
    {
        unsigned int cluster_id = mvEleEdgeClusters[i].cluster_id;
        mmClusterID2index[cluster_id] = i;
    }


    // * STEP-3 从 mvElementEdges 中删掉这些边缘
    //-- 从后往前删，不破坏索引
    std::sort(indicesToDelete.begin(), indicesToDelete.end(), std::greater<int>());
    for(int idx : indicesToDelete) {
        mvElementEdges.erase(mvElementEdges.begin() + idx);
    }
    //-- 重新索引 mvElementEdges
    mmElementID2index.clear();
    for(size_t i = 0; i < mvElementEdges.size(); ++i)
    {
        elementEdge& ele = mvElementEdges[i];
        mmElementID2index[ele.element_id] = i;
    }


    // * STEP-4 移除关键帧，并清理关键帧与局部地图有关的信息
    pKF_del->mmEdgeIndex2ElementEdgeID.clear();
    pKF_del->mmMapAssociations.clear();

    //-- 移除第一个帧
    mvKeyFrames.erase(mvKeyFrames.begin());
    mmKFID2KFindex.clear();
    for(int i = 0; i < mvKeyFrames.size(); ++i)
    {
        int ckf_id = mvKeyFrames[i]->KF_ID;
        mmKFID2KFindex[ckf_id] = i;
    }

    //-- 修改local_map的参考坐标位置为当前首帧的位姿
    // assert(mvKeyFrames.size()>0);
    // T_ref = mvKeyFrames[0]->KF_pose_g;
}

pcl::PointXYZ localMap::calcCloudCentroid(const pcl::PointCloud<pcl::PointXYZ>& pointCloud, pcl::PointXYZ current)
{
    pcl::PointXYZ NaN(0,0,0);
    pcl::PointXYZ centroid = current;
    // //-- 防止current点自己深度不好，不考虑current点
    // centroid.x = 0;
    // centroid.y = 0;
    // centroid.z = 0;
    for (const auto& point : pointCloud.points) {  
        centroid.x += point.x;  
        centroid.y += point.y;  
        centroid.z += point.z;  
    }
    
    int numPoints = pointCloud.points.size() + 1;
    if(numPoints == 1){
        return NaN;
    }
    centroid.x /= numPoints;
    centroid.y /= numPoints;
    centroid.z /= numPoints;
    return centroid;  
}

void localMap::getMergedCluster(const std::vector<pcl::PointCloud<pcl::PointXYZ>>& clusterCloud,
                                  pcl::PointCloud<pcl::PointXYZ>& mergedCloud)
{
    //-- 寻找最长的一条边缘
    int maxLength = 0;
    int maxIndex = -1;
    for (int i = 0; i < clusterCloud.size(); ++i){
        if (clusterCloud[i].size() > maxLength){
            maxLength = clusterCloud[i].size();
            maxIndex = i;
        }
    }

    //-- 对最长的边缘进行采样
    pcl::PointCloud<pcl::PointXYZ> sampled_longest_cloud;
    const int sampleBias = 3;
    const auto& longestCloud = clusterCloud[maxIndex]; 
    const size_t cloudSize = longestCloud.size();

    sampled_longest_cloud.reserve(cloudSize / sampleBias + 2);
    sampled_longest_cloud.push_back(longestCloud[0]);  // 第一个点
    for(size_t i = sampleBias; i < cloudSize - 1; i += sampleBias) {
        sampled_longest_cloud.push_back(longestCloud[i]);
    }
    if(cloudSize > 1) {  // 最后一个点
        sampled_longest_cloud.push_back(longestCloud.back());
    }


    //-- 对最长的边缘作为参考边缘，建立KDtree
    pcl::PointCloud<pcl::PointXYZ>::Ptr p_ref_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    *p_ref_cloud = sampled_longest_cloud;
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;  
    kdtree.setInputCloud (p_ref_cloud);

    //-- 最长的边的端点，用来判断新加入的边插在哪边
    pcl::PointXYZ end_point_1 = p_ref_cloud->points[0];
    pcl::PointXYZ end_point_2 = p_ref_cloud->points.back();

    //-- 存储与当前采样边缘有关联的其他点的信息
    std::vector<pcl::PointCloud<pcl::PointXYZ>> associatedClouds;
    associatedClouds.resize(p_ref_cloud->points.size());
    
    //-- 通过垂足关系找到与这条边缘关联的其他的关联部分
    for(int i = 0; i < clusterCloud.size(); ++i){
        std::vector<bool> isAssociated;
        if(i == maxIndex){
            continue;
        }
        //-- 遍历当前边缘，对每个边缘点计算一个垂足
        for(int j = 0; j < clusterCloud[i].size(); ++j){
            //-- 最近邻搜索得到垂线的基准点
            pcl::PointXYZ query = clusterCloud[i][j];
            int K = 1;
            std::vector<int> pointIdxNKNSearch(K);
            std::vector<float> pointNKNSquaredDistance(K);
            kdtree.nearestKSearch(query, K, pointIdxNKNSearch, pointNKNSquaredDistance);
            int nearestIndex = pointIdxNKNSearch[0];
            float nearestDistance = pointNKNSquaredDistance[0];
            //-- 根据左右方向得到垂线端点
            int index_front = nearestIndex-1>=0 ? nearestIndex-1 : 0;
            int index_rear = nearestIndex+1<p_ref_cloud->points.size() ? nearestIndex+1 : p_ref_cloud->points.size()-1;
            pcl::PointXYZ point_a = p_ref_cloud->points[index_front];
            pcl::PointXYZ point_b = p_ref_cloud->points[index_rear];
            //-- 计算垂足
            Eigen::Vector3d bias;
            bias(0) = point_a.x - point_b.x;
            bias(1) = point_a.y - point_b.y;
            bias(2) = point_a.z - point_b.z;
            Eigen::Vector3d connect;
            connect(0) = point_b.x - query.x;
            connect(1) = point_b.y - query.y;
            connect(2) = point_b.z - query.z;
            double length = - connect.dot(bias)/(bias.norm() * bias.norm());
            pcl::PointXYZ foot;
            foot.x = length*bias(0) + point_b.x;
            foot.y = length*bias(1) + point_b.y;
            foot.z = length*bias(2) + point_b.z;
            //-- 判断垂足在不在两点中间
            if(length < 0 || length > 1){//-- 垂足不在两点之间，说明点关联偏了
                isAssociated.push_back(false);
            }else{//-- 垂足中正
                isAssociated.push_back(true);
                //-- 根据关联关系，更新与参考边缘关联的点云，一定距离阈值内的才要更新
                if(nearestDistance < 0.03){
                    int asso_idx = nearestIndex;
                    float nearest_foot_dist = 0;
                    pcl::PointXYZ point_n = p_ref_cloud->points[nearestIndex];
                    //-- 判断垂足离point_a, point_b, point_n谁最近
                    double distance_1 = pcl::euclideanDistance(foot, point_n);
                    double distance_2 = pcl::euclideanDistance(foot, point_a);
                    double distance_3 = pcl::euclideanDistance(foot, point_b);
                    if(distance_1 <= distance_2 && distance_1 <= distance_3){
                        asso_idx = nearestIndex;
                        nearest_foot_dist = distance_1;
                    }else if(distance_2 <= distance_1 && distance_2 <= distance_3){
                        asso_idx = index_front;
                        nearest_foot_dist = distance_2;
                    }else if(distance_2 <= distance_1 && distance_2 <= distance_3){
                        asso_idx = index_rear;
                        nearest_foot_dist = distance_3;
                    }

                    if(nearest_foot_dist < 0.02){
                        associatedClouds[asso_idx].points.push_back(query);
                    }
                }
            }
        }

        // //-- 寻找当前边缘false关联最长的一段
        // int maxLength = 0, currentLength = 0;
        // int start_index = -1, end_index = -1;
        // int final_start_index;
        // for(int j = 0; j < isAssociated.size(); ++j){
        //     bool b = isAssociated[j];
        //     if(b == false){
        //         //-- 如果首次出现false或由true变为false, 则记录这个index
        //         if(currentLength == 0) start_index = j;
        //         currentLength++;
        //     }else{
        //         if(currentLength > maxLength){
        //             maxLength = currentLength;
        //             final_start_index = start_index;
        //             end_index = j;
        //         }
        //         currentLength = 0;
        //     }
        // }
        // //-- 遍历完一遍，如果结尾全是false且最大则调整之
        // if(currentLength > maxLength){
        //     maxLength = currentLength;
        //     final_start_index = start_index;
        //     end_index = isAssociated.size();
        // }
        // //-- 现在final_start_index到end_index-1中间这一部分是需要被添加的部分
        // int false_length = end_index - final_start_index;
        // //-- 太短的不要
        // if(false_length < 5){
        //     continue;
        // }

        // //-- 整理出多出的这一段
        // pcl::PointCloud<pcl::PointXYZ> addCloud;
        // for(int j = final_start_index; j < end_index; ++j){
        //     if(j == final_start_index || j == end_index - 1 || (j-final_start_index)%3==0){
        //         addCloud.push_back(clusterCloud[i].points[j]);
        //     }
        // }

        // //-- 多出的这一段的关联关系同样需要修改
        // std::vector<pcl::PointCloud<pcl::PointXYZ>> add_associatedClouds;
        // add_associatedClouds.resize(addCloud.size());

        // //-- 判断这一段应该插在哪里
        // pcl::PointXYZ end_point_3 = addCloud.points[0];
        // pcl::PointXYZ end_point_4 = addCloud.points.back();
        // double dst1 = sqrt((end_point_3.x-end_point_1.x)*(end_point_3.x-end_point_1.x)
        //                   +(end_point_3.y-end_point_1.y)*(end_point_3.y-end_point_1.y)
        //                   +(end_point_3.z-end_point_1.z)*(end_point_3.z-end_point_1.z));
        // double dst2 = sqrt((end_point_3.x-end_point_2.x)*(end_point_3.x-end_point_2.x)
        //                   +(end_point_3.y-end_point_2.y)*(end_point_3.y-end_point_2.y)
        //                   +(end_point_3.z-end_point_2.z)*(end_point_3.z-end_point_2.z));
        // double dst3 = sqrt((end_point_4.x-end_point_1.x)*(end_point_4.x-end_point_1.x)
        //                   +(end_point_4.y-end_point_1.y)*(end_point_4.y-end_point_1.y)
        //                   +(end_point_4.z-end_point_1.z)*(end_point_4.z-end_point_1.z));
        // double dst4 = sqrt((end_point_4.x-end_point_2.x)*(end_point_4.x-end_point_2.x)
        //                   +(end_point_4.y-end_point_2.y)*(end_point_4.y-end_point_2.y)
        //                   +(end_point_4.z-end_point_2.z)*(end_point_4.z-end_point_2.z));
        // pcl::PointCloud<pcl::PointXYZ> cloud_merge = *p_ref_cloud;
        // if(dst1 <= dst2 && dst1 <= dst3 && dst1 <= dst4){
        //     //-- 插入顺序为4--3|1--2
        //     std::reverse(addCloud.begin(), addCloud.end());
        //     cloud_merge = addCloud + cloud_merge;
        //     associatedClouds.insert(associatedClouds.begin(), 
        //                             add_associatedClouds.begin(), add_associatedClouds.end());
        // }else if(dst2 <= dst1 && dst2 <= dst3 && dst2 <= dst4){
        //     //-- 插入顺序为1--2|3--4
        //     cloud_merge = cloud_merge + addCloud;
        //     associatedClouds.insert(associatedClouds.end(), 
        //                             add_associatedClouds.begin(), add_associatedClouds.end());
        // }else if(dst3 <= dst1 && dst3 <= dst2 && dst3 <= dst4){
        //     //-- 插入顺序为3--4|1--2
        //     cloud_merge = addCloud + cloud_merge;
        //     associatedClouds.insert(associatedClouds.begin(), 
        //                             add_associatedClouds.begin(), add_associatedClouds.end());
        // }else if(dst4 <= dst1 && dst4 <= dst2 && dst4 <= dst3){
        //     //-- 插入顺序为1--2|4--3
        //     std::reverse(addCloud.begin(), addCloud.end());
        //     cloud_merge = cloud_merge + addCloud;
        //     associatedClouds.insert(associatedClouds.end(), 
        //                             add_associatedClouds.begin(), add_associatedClouds.end());
        // }

        // //-- 更新kdtree
        // *p_ref_cloud = cloud_merge;
        // pcl::KdTreeFLANN<pcl::PointXYZ> kdtree_new;  
        // kdtree_new.setInputCloud (p_ref_cloud);
        // kdtree = kdtree_new;
        // end_point_1 = p_ref_cloud->points[0];
        // end_point_2 = p_ref_cloud->points.back();

        // //-- 更新isAssociated
        // for(int j = final_start_index; j < end_index; ++j){
        //     isAssociated[j] = true;
        // }

    } 
    // //-- 计算关联的质心
    pcl::PointCloud<pcl::PointXYZ> cloud_adjust;
    pcl::PointXYZ NaN(0,0,0);
    for(int cnt = 0; cnt < p_ref_cloud->points.size(); ++cnt)
    {
        pcl::PointXYZ centroid = calcCloudCentroid(associatedClouds[cnt], p_ref_cloud->points[cnt]);
        if(centroid.x == 0 && centroid.y == 0 && centroid.z == 0){
            continue; //-- 遇到没有关联的NaN的点就不管这个点
        }
        cloud_adjust.push_back(centroid);
        //cloud_adjust.push_back(p_ref_cloud->points[cnt]);
    }

    mergedCloud = cloud_adjust;
    return;
}

void localMap::clustersFitting3D()
{
    //-- 遍历所有的cluster
    for(size_t i = 0; i < mvEleEdgeClusters.size(); ++i)
    {
        
        elementEdgeCluster& cluster = mvEleEdgeClusters[i];
        
        const std::vector<unsigned int>& ele_ids = cluster.mvElementEdgeIDs;
        if(ele_ids.size() < mvKeyFrames.size()/2)continue;

        std::vector<int> ele_indices(ele_ids.size(), -1);
        for(size_t j = 0; j < ele_ids.size(); ++j)
        {
            ele_indices[j] = mmElementID2index[ele_ids[j]];
        }

        std::vector<pcl::PointCloud<pcl::PointXYZ>> clusterCloud;
        clusterCloud.reserve(ele_indices.size());
        //-- 构造
        int N = ele_indices.size();
        for(size_t j = 0; j < N; ++j)
        {
            elementEdge& ele_edge = mvElementEdges.at(ele_indices[j]);
            int kf_id = ele_edge.kf_id;
            int edge_idx = ele_edge.kf_edge_idx;
            int kf_idx = mmKFID2KFindex.at(kf_id);
            Edge& edge = mvKeyFrames[kf_idx]->mvEdges[edge_idx];

            Sophus::SE3d T_ref_cur = mvKeyFrames[kf_idx]->KF_pose_g;
            
            pcl::PointCloud<pcl::PointXYZ> cloud;
            for(size_t k = 0; k < edge.mvPoints.size(); ++k)
            {
                const orderedEdgePoint& pt = edge.mvPoints[k];
                //-- 计算3D坐标
                Eigen::Vector3d point_3d(pt.x_3d,pt.y_3d,pt.z_3d);
                //-- 重投影得到新的投影点
                point_3d = T_ref_cur * point_3d;
                pcl::PointXYZ point(point_3d.x(),point_3d.y(),point_3d.z());
                cloud.points.push_back(point);
            }
            clusterCloud.push_back(cloud);
        }
        //-- 在 T_ref 坐标系下的 merged_cloud;
        pcl::PointCloud<pcl::PointXYZ> merged_cloud;
        getMergedCluster(clusterCloud, merged_cloud);

        //-- 将 pcl::PointCloud 类型转为 opencv 的点云类型
        std::vector<cv::Point3d> result;
        result.reserve(merged_cloud.size());
        
        const pcl::PointXYZ* data = merged_cloud.points.data();
        const size_t size = merged_cloud.size();
        for (size_t i = 0; i < size; ++i) {
            result.emplace_back(data[i].x, data[i].y, data[i].z);
        }
        //-- 把 merged 的结果存放在 cluster 中
        cluster.mvMergedCloud_ref = std::move(result);
        cluster.mbMerged = true; 

    }
}




void localMap::clusterFittingProjection()
{
    //-- 重新构造就先清理掉
    for(size_t i = 0; i < mvEleEdgeClusters.size(); ++i)
    {
        elementEdgeCluster& cluster = mvEleEdgeClusters[i];
        cluster.mbMerged = false;
        cluster.mvMergedCloud_ref.clear();
        cluster.dist_thres = -1.0;
    }

    //-- 遍历所有的cluster
    for(size_t i = 0; i < mvEleEdgeClusters.size(); ++i)
    {
        
        elementEdgeCluster& cluster = mvEleEdgeClusters[i];
        
        const std::vector<unsigned int>& ele_ids = cluster.mvElementEdgeIDs;
        if(ele_ids.size() < 5)continue;

        std::vector<int> ele_indices(ele_ids.size(), -1);
        for(size_t j = 0; j < ele_ids.size(); ++j)
        {
            ele_indices[j] = mmElementID2index[ele_ids[j]];
        }

        std::vector<elementEdge> clusterElement;
        std::vector<int> kf_indices;
        clusterElement.reserve(ele_indices.size());
        kf_indices.reserve(ele_indices.size());
        //-- 构造
        int N = ele_indices.size();
        for(size_t j = 0; j < N; ++j)
        {
            elementEdge& ele_edge = mvElementEdges.at(ele_indices[j]);
            int kf_id = ele_edge.kf_id;
            int kf_idx = mmKFID2KFindex.at(kf_id);
            clusterElement.push_back(ele_edge);
            kf_indices.push_back(kf_idx);
        }
        //-- 在 T_ref 坐标系下的 merged_cloud;
        std::vector<cv::Point3d> merged_cloud;
        double dist_thres;
        std::vector<int> involved_elements;
        //-- featureMerger::getMergedClusterProjection(clusterElement, kf_indices, mvKeyFrames, merged_cloud, involved_elements);
        // featureMerger::getMergedClusterIncremental(clusterElement, kf_indices, mvKeyFrames, merged_cloud, involved_elements);
        featureMerger::getMergedClusterIterative(clusterElement, kf_indices, mvKeyFrames, merged_cloud, dist_thres, involved_elements);
        //-- 把 merged 的结果存放在 cluster 中
        cluster.mvMergedCloud_ref = std::move(merged_cloud);
        cluster.mvInvolvedLocalMapElementIndices = std::move(involved_elements);
        cluster.mbMerged = true; 
        cluster.dist_thres = dist_thres;

    }

    assignWeights();
}

void localMap::assignWeights()
{
    std::vector<double> dist_thres_values;
    for (auto& cluster : mvEleEdgeClusters)
    {
        if (cluster.mbMerged == false) continue;
        dist_thres_values.push_back(cluster.dist_thres);
        cluster.weightBA = 1.0;
    }

    // 1. 检查输入是否为空
    if (dist_thres_values.empty()) {
        return;
    }

    // 2. 创建排序后的副本（从大到小）
    std::vector<double> sorted_values = dist_thres_values;
    std::sort(sorted_values.begin(), sorted_values.end(), std::greater<double>());

    // 3. 计算平均值
    double sum = std::accumulate(sorted_values.begin(), sorted_values.end(), 0.0);
    double mean = sum / sorted_values.size();

    for (auto& cluster : mvEleEdgeClusters)
    {
        if (cluster.mbMerged == false) continue;
        double curr_dist = cluster.dist_thres;

        // 4. 如果dist小于等于平均值，权重为1
        if (curr_dist <= mean) {
            cluster.weightBA = 1.0;
            continue;
        }

        // 5. 计算dist在排序列表中的相对位置（百分位）
        auto it = std::lower_bound(sorted_values.begin(), sorted_values.end(), curr_dist, std::greater<double>());
        double rank = std::distance(sorted_values.begin(), it);
        double percentile = rank / sorted_values.size();

        // 6. 设计权重函数：满足dist越大权重越小，衰减速度逐渐变慢
        // 使用指数衰减函数，但调整衰减速率
        double normalized_dist = (curr_dist - mean) / (sorted_values.front() - mean);
        double weight = std::exp(-normalized_dist * (1.0 + percentile));
        cluster.weightBA = weight;
    }
}


void localMap::getAssoFrameMergeEdge(int kf_id_dst, std::vector<match3d_2d>& matches, std::vector<double>& weights)
{
    matches.clear();
    weights.clear();

    for (const auto& cluster : mvEleEdgeClusters)
    {
        if (cluster.mbMerged == false) continue;
        if (cluster.weightBA <= 0) continue;
        
        std::vector<cv::Point3d> merged_cloud = cluster.mvMergedCloud_ref;
        std::vector<elementEdge> associated_ele_edges;

        //-- 每个cluster的构成边缘
        std::vector<int> involved_ele_indices = cluster.mvInvolvedLocalMapElementIndices;
        if(involved_ele_indices.empty())
        {
            continue;
        }

        for (size_t i = 0; i < involved_ele_indices.size(); ++i)
        {
            int idx = involved_ele_indices[i];
            unsigned int ele_id = cluster.mvElementEdgeIDs[idx];

            int ele_idx = mmElementID2index.at(ele_id);
            elementEdge& ele_edge = mvElementEdges[ele_idx];
            int kf_id = ele_edge.kf_id;
            int kf_edge_index = ele_edge.kf_edge_idx;

            if(kf_id == kf_id_dst)
            {
                associated_ele_edges.push_back(ele_edge);
            }
        }

        if( !associated_ele_edges.empty())
        {
            match3d_2d match;
            match.first = merged_cloud;
            match.second = associated_ele_edges;
            
            double weight = cluster.weightBA;

            matches.push_back(match);
            weights.push_back(weight);
        }
    }
}

