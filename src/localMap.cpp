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
        pKF_0->mmMapEdge2EdgeElement[i] = ele.element_id;
    }

    //-- 初始化整个局部地图的基准坐标系
    T_ref = pKF_0->KF_pose_g;

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
            unsigned int map_id_1 = pKF_0->mmMapEdge2EdgeElement.at(root_idx);
            unsigned int map_id_2 = pKF_0->mmMapEdge2EdgeElement.at(edge_idx);

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
        if(pKF_0->mmMapEdge2EdgeElement.find(ref_idx)==pKF_0->mmMapEdge2EdgeElement.end()) continue;

        int size = mvElementEdges.size();
        elementEdge ele(pKF_1->KF_ID, cur_idx);
        //-- 与当前帧的该边缘关联的参考帧的地图id
        ele.union_id = pKF_0->mmMapEdge2EdgeElement[ref_idx];

        mvElementEdges.push_back(ele);
        mmElementID2index[ele.element_id] = mvElementEdges.size()-1;

        pKF_1->mmMapEdge2EdgeElement[cur_idx] = ele.element_id;
    }

    // * STEP-3 将 pKF_0 中不能与 pKF_1 关联的边缘移除出局部地图
    for (auto it = mvElementEdges.begin(); it != mvElementEdges.end(); ) {
        if(it->kf_id == pKF_0->KF_ID && !isAssociated[it->kf_edge_idx]) 
        {
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


}

void localMap::initWithRefFrame(KeyFramePtr frame_ref)
{
    Eigen::Matrix4d E = Eigen::MatrixXd::Identity(4,4);
    int frame_id = frame_ref->KF_ID;
    for(int i = 0; i < frame_ref->mvEdges.size(); ++i)
    {
        //-- 关键帧中的边缘都是有效边缘
        int edge_idx = i;
        elementEdge ele(frame_id, edge_idx);
        //-- 局部地图添加当前的边缘元素
        mvElementEdges.push_back(ele);
        //-- 关键帧索引当前的局部地图
        frame_ref->mmMapEdge2EdgeElement[i] = mvElementEdges.size()-1;
    }
    //-- 局部地图添加当前关键帧
    mvKeyFrames.emplace_back(frame_ref);

    //-- 初始化整个局部地图的基准坐标系
    T_ref = frame_ref->KF_pose_g;
}

//-- 输入 curr_id 是当前 elementEdge 的 element_id
unsigned int localMap::findRoot(unsigned int curr_id)
{
    //-- 与当前帧同类的根节点
    int curr_idx = mmElementID2index[curr_id];
    int union_id = mvElementEdges[curr_idx].union_id;
    if(union_id == -1){//-- 如果当前的curr_id的根节点就是-1，说明这个就是独立的根节点
        return curr_id;
    }else{//-- 如果不是，就递归找根节点
        return findRoot(union_id);
    }
}

void localMap::pruningMap(){
    for(int i = 0; i < mvElementEdges.size(); ++i){
        int root_idx = findRoot(i);
        mvElementEdges[i].union_id = root_idx;
    }
}

void localMap::generateLabelMap()
{
    mmElementEdgeMap.clear();
    std::cout<<mvElementEdges.size()<<std::endl;
    for(int i = 0; i < mvElementEdges.size(); ++i){
        unsigned int curr_id = mvElementEdges[i].element_id;
        unsigned int root_id = findRoot(curr_id);
        //std::cout<<root_idx<<std::endl;
        //-- root_idx是0的时候会有BUG, 需要检查
        if(mmElementEdgeMap.find(root_id) == mmElementEdgeMap.end()){
            //-- root_idx并不存在，则创建一个新key
            std::vector<int> labelList;
            labelList.push_back(i);
            mmElementEdgeMap[root_id] = labelList;
        }else{
            //-- root_idx存在，则更新list
            mmElementEdgeMap[root_id].push_back(i);
        }
    }
    std::cout<<"finish generating"<<std::endl;
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
        std::vector<int> associated_edges = frame_ref->edgeWiseCorrespondenceReproject(query_edge,T_ref_cur);
        
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

void localMap::addNewFrame(KeyFramePtr frame_cur)
{
    //-- 知道当前帧哪些帧能与参考帧关联，哪些帧不行
    int N = frame_cur->mvEdges.size();
    //-- 当前帧的frame与地图元素的映射
    std::map<int, int> curFrame2MapElement;
    //-- 当前帧的关联列表，能与先前任一参考帧关联的都会被设为true
    std::vector<bool> isAssociated(N, false);

    for(int i = 0; i < mvKeyFrames.size(); ++i){
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
        std::map<int, std::vector<int>> labelRefMap;
        std::unordered_map<int, int> edgeLabelMap = res.second; //-- 边缘-label map
        std::unordered_map<int, int> curRefMap = res.first;     //-- 当前帧-参考帧 map

        //-- 参考帧内部进行并查集合并（参考帧自己的边缘是同一个边缘）
        for(const auto &pair : edgeLabelMap){
            //-- 当前的参考帧可关联边缘的label
            int label = pair.second;
            //-- 当前的参考帧可关联边缘的index
            int edge_idx = pair.first;
            if(labelRefMap.find(label) == labelRefMap.end()){
                std::vector<int> tmp;
                tmp.push_back(edge_idx);
                labelRefMap[label] = tmp;
            }else{
                labelRefMap[label].push_back(edge_idx);
                //-- 参考帧中的root id
                int cluster_id_1 = labelRefMap[label].front();
                //-- 参考帧中的同类id
                int cluster_id_2 = edge_idx;
                //-- 由帧边缘ID对应到地图的边缘ID，并使用并查集进行合并这些边缘
                int map_idx_1 = mvKeyFrames[i]->mmMapEdge2EdgeElement[cluster_id_1];
                int map_idx_2 = mvKeyFrames[i]->mmMapEdge2EdgeElement[cluster_id_2];
                //-- 并查集的归并
                mvElementEdges[map_idx_2].union_id = findRoot(map_idx_1);
            }
        }

        //-- 遍历当前帧的关联列表，将能够关联的帧加入Map中，能关联的当前帧边缘一定是有效的
        for(const auto& pair: curRefMap){
            int cur_id = pair.first;
            //-- 如果当前边缘已经被其他参考帧关联了，那跳过这个边缘
            if(isAssociated[cur_id]==true) continue;
            int ref_id = pair.second;
            int label_id = edgeLabelMap[ref_id];

            //-- 如果被关联到的参考帧是无效帧，在frame2MapElement中找不到，则跳过
            if(mvKeyFrames[i]->mmMapEdge2EdgeElement.find(ref_id)==mvKeyFrames[i]->mmMapEdge2EdgeElement.end()) continue;

            //-- 对于已经关联了的帧，创建一个elementMap
            isAssociated[cur_id] = true; //-- 表示这个帧已经被关联了
            int size = mvElementEdges.size();
            elementEdge ele(frame_cur->KF_ID, cur_id);
            //-- 与当前帧的该边缘关联的参考帧的地图id
            int map_idx_1 = mvKeyFrames[i]->mmMapEdge2EdgeElement[ref_id];
            ele.union_id = map_idx_1;
            mvElementEdges.push_back(ele);
            curFrame2MapElement[cur_id] = mvElementEdges.size()-1;
        }
    }
    //-- 遍历当前帧未被关联的边缘，将其加入到地图中，这一步可能存在无效边缘，需要剔除
    for(int i = 0; i < frame_cur->mvEdges.size(); ++i){
        if(isAssociated[i]== true) continue;
        //-- 对于未被关联的边缘，加入地图中
        int size = mvElementEdges.size();
        elementEdge ele(frame_cur->KF_ID, i);
        mvElementEdges.push_back(ele);
        curFrame2MapElement[i] = mvElementEdges.size()-1;
    }
    frame_cur->mmMapEdge2EdgeElement = curFrame2MapElement;
    //-- 将当前帧添加入Map
    mvKeyFrames.push_back(frame_cur);
}

