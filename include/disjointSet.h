#ifndef DISJOINT_SET_H
#define DISJOINT_SET_H

#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <memory> //-- 使用智能指针（shared_ptr）需要引用这个头文件

// #define __DEBUG_DISJOINTSET__

class DisjointSet
{
  private:
    std::vector<int> parent; //index list for label merging
    std::vector<int> rank; // height of the disjoint set tree.

  public:
    //defination of intelligent pointer
    typedef std::shared_ptr<DisjointSet>       Ptr;
    typedef const std::shared_ptr<DisjointSet> ConstPtr;
    DisjointSet(int max_size) : parent(std::vector<int>(max_size)),
                                rank(std::vector<int>(max_size, 0)){
        for (int i = 0; i < max_size; ++i) parent[i] = i;
    }
    //iterated find methods
    int find(int x){
        if(x == parent[x]){
            return x;
        }else{
            parent[x] = find(parent[x]);
            return parent[x];
        }//return x == parent[x] ? x : (parent[x] = find(parent[x]));
    }

    //pruning so that the whole set will have lower rank.
    void pruningSet(){
        for(size_t i = 0; i < parent.size(); ++i){
            parent[i]=find(parent[i]);
            rank[parent[i]] = 1;
        }
    }

    //merge x1,x2 to be one group
    void to_union(int x1, int x2){
        int f1 = find(x1);
        int f2 = find(x2);
        if (rank[f1] > rank[f2]){
            parent[f2] = f1;
            #ifdef __DEBUG_DISJOINTSET__
                std::cout<<"parent \033[32m"<<f2<<" \033[0mis now \033[32m"<<f1<<"\033[0m"<<std::endl;
            #endif
        }else{
            parent[f1] = f2;
            #ifdef __DEBUG_DISJOINTSET__
                std::cout<<"parent \033[32m"<<f1<<" \033[0mis now \033[32m"<<f2<<"\033[0m"<<std::endl;
            #endif
            if (rank[f1] == rank[f2]) ++rank[f2];
        }
    }

    bool is_same(int e1, int e2){
        return find(e1) == find(e2);
    }

    void show(){
        std::cout<<"\033[34m#####################################\033[0m"<<std::endl;
        for(size_t i = 0; i < parent.size(); ++i){
            std::cout<<i<<"\033[33m <> \033[0m"<<parent[i]<<std::endl;
        }
    }

};

#endif
