#include "elementEdge.h"

using namespace edge_map;
// 静态counter 初始化
unsigned int elementEdge::id_counter = 0;

std::mt19937 elementEdgeCluster::rng;
std::uniform_int_distribution<unsigned char> elementEdgeCluster::dist(0, 255);