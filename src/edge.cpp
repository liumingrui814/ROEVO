#include "edge.h"

Edge::Edge(int ID, std::vector<orderedEdgePoint> list){
    edge_ID = ID;
    for(int i = 0; i < list.size(); ++i){
        list[i].frame_edge_ID = ID;
        this->push_back(list[i]);
    }
}

void Edge::push_back(orderedEdgePoint pt){
    mvPoints.push_back(pt);
}

void Edge::samplingEdgeUniform(int bias){
    mvSampledEdgeIndex.clear();

    //-- 进行采样
    for(int i = 0; i < mvPoints.size(); ++i){
        if(i % bias == 0){
            mvSampledEdgeIndex.push_back(i);
        }
    }
}

bool Edge::isAssociatedWith(Edge query_edge){
    return true;
}

//-- 根据所有边缘点的可见性分数计算边缘本身的可见性分数
void Edge::calcEdgeScoreViz()
{
    //-- 由于大量随机噪声的存在，用中位数比较合适
    std::vector<double> score;
    double total_score;
    for(int i = 0; i < mvPoints.size(); ++i){
        score.push_back(mvPoints[i].score_visible);
        total_score += mvPoints[i].score_visible;
    }
    std::sort(score.begin(), score.end());
    double medianValue;
    if(score.size()%2==0){
        medianValue = 1.0/2.0 * (score[score.size()/2] + score[score.size()/2 - 1]);
    }else{
        medianValue = score[score.size()/2];
    }
    //-- 平均值作为score
    //-- mVisScore = total_score / mvPoints.size();
    //-- 中位值作为score
    mVisScore = medianValue;
}