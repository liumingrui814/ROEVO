#ifndef KF_FILE_H
#define KF_FILE_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip> // 注意包含这个头文件, 有效数字需要

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>

namespace kf_file{

//-- 输入的列表是关键帧对应的rgb图片的名字，以避免保存时间戳可能存在的精度损失的问题
//-- 存储索引-关键帧rgb图片名字的序列，用于支持回环边的构建以及后续的所有图优化任务
void saveKFstampFileIndex(std::string file_name, const std::vector<std::string>& rgb_name_list)
{
    std::ofstream file(file_name, std::ios::out);
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << file_name << std::endl;
        return;
    }

    int index = 0;
    for (const auto& line : rgb_name_list) {
        file << index <<" "<<line << std::endl;
        index++;
    }
    file.close();
}

void saveLoopPairWithScore(std::string file_name,
                           std::vector<std::pair<unsigned int , unsigned int>> loopPairs,
                           std::vector<float> loopScores)
{
    std::ofstream file(file_name, std::ios::out);
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << file_name << std::endl;
        return;
    }

    int index = 0;
    if(loopPairs.size() != loopScores.size()){
        std::cout<<"\033[31m [ERROR SAVING PAIR] \033[0m"<<": loop pairs and scores size no match!"<<std::endl;
    }
    for (int i = 0; i < loopPairs.size(); ++i) {
        file << loopPairs[i].first <<" "
             << loopPairs[i].second << " " 
             << loopScores[i] << std::endl;
        index++;
    }
    file.close();
}

void saveLoopEdges(std::string file_name,
                   std::vector<std::pair<unsigned int , unsigned int>> loopPairs,
                   std::vector<float> loopScores,
                   std::vector<Eigen::Matrix4d> trans)
{
    std::ofstream file(file_name, std::ios::out);
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << file_name << std::endl;
        return;
    }

    int index = 0;
    if(loopPairs.size() != loopScores.size()){
        std::cout<<"\033[31m [ERROR SAVING PAIR] \033[0m"<<": loop pairs and scores size no match!"<<std::endl;
    }
    for (int i = 0; i < loopPairs.size(); ++i) {
        Eigen::Vector3d translation = trans[i].block(0,3,3,1);
        Eigen::Matrix3d rotation = trans[i].block(0,0,3,3);
        Eigen::Quaterniond q(rotation);
        file << loopPairs[i].first <<" "
             << loopPairs[i].second << " " 
             << loopScores[i] << " "
             << translation(0) <<" "
             << translation(1) <<" "
             << translation(2) <<" "
             << q.x() <<" "
             << q.y() <<" "
             << q.z() <<" "
             << q.w() <<std::endl;
        index++;
    }
    file.close();
}

void saveGlobalLoopPairs(std::string file_name,
                         std::vector<std::pair<unsigned int , unsigned int>> loopPairs)
{
    std::ofstream file(file_name, std::ios::out);
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << file_name << std::endl;
        return;
    }

    int index = 0;
    for (int i = 0; i < loopPairs.size(); ++i) {
        file << loopPairs[i].first <<" "
             << loopPairs[i].second << std::endl;
        index++;
    }
    file.close();
}

//-- 读入带有index的关键帧序列，index与关键帧的图像名序列列表在索引上对应
void loadKFstampFileIndex(std::string file_name, 
                          std::vector<std::string>& stamp_list,
                          std::vector<int>& index_list)
{
    stamp_list.clear();
    index_list.clear();
    std::ifstream file(file_name);
    
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << file_name << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string file_name;
        int index;
        //-- 读入一行的第一个字，即关键帧的整数索引
        iss >> index >> file_name;
        index_list.push_back(index);
        stamp_list.push_back(file_name);
    }

    file.close();
}


//-- 读入回环边以及回环边的分数，存储在列表中
void loadLoopPairsWithScores(std::string file_name, 
                        std::vector<std::pair<unsigned int , unsigned int>>& loopPairs,
                        std::vector<float>& loopScores)
{
    loopPairs.clear();
    loopScores.clear();
    std::ifstream file(file_name);
    
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << file_name << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        int kf_index_1;
        int kf_index_2;
        float score;
        //-- 读入一行的第一个字，即关键帧的整数索引
        iss >> kf_index_1 >> kf_index_2 >> score;
        std::pair<unsigned int , unsigned int> res_pair;
        res_pair.first = kf_index_1;
        res_pair.second = kf_index_2;
        loopPairs.push_back(res_pair);
        loopScores.push_back(score);
    }

    file.close();
}


//-- 读取数据集后，根据一个关键帧的rgb名字找到对应的时间戳
std::vector<double> locateStamps(const std::vector<std::string>& rgb_kf_list,  
                   const std::vector<double>& stamp_list, 
                   const std::vector<std::string>& rgb_list)
{
    std::vector<double> stamps_kf(rgb_kf_list.size(), 0);
    for(int i = 0; i < rgb_kf_list.size(); ++i){
        std::string rgb_query = rgb_kf_list[i];
        for(int j = 0; j < rgb_list.size(); ++j){
            if(rgb_list[j] == rgb_query){
                stamps_kf[i] = stamp_list[j];
                break;
            }
        }
    }
    return stamps_kf;
}

void loadLoopEdges(std::string file_name,
                   std::vector<std::pair<unsigned int , unsigned int>>& loopPairs,
                   std::vector<float>& loopScores,
                   std::vector<Eigen::Matrix4d>& trans)
{
    std::cout<<"load loop edges from file "<<file_name<<" ..."<<std::endl;
    loopPairs.clear();
    loopScores.clear();
    trans.clear();

    std::ifstream file(file_name, std::ios::in);
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << file_name << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        // std::cout<<line<<std::endl;
        std::istringstream iss(line);
        int kf_index_1;
        int kf_index_2;
        float score;
        double tx, ty, tz, qx, qy, qz, qw;
        //-- 读入一行的第一个字，即关键帧的整数索引
        iss >> kf_index_1 >> kf_index_2 >> score>>tx>>ty>>tz>>qx>>qy>>qz>>qw;
        // std::cout<<tx<<","<<ty<<","<<tz<<","<<
        //            qx<<","<<qy<<","<<qz<<","<<qw<<std::endl;
        std::pair<unsigned int , unsigned int> res_pair;
        res_pair.first = kf_index_1;
        res_pair.second = kf_index_2;
        Eigen::Quaterniond q(qw, qx, qy, qz);
        Eigen::Vector3d t(tx, ty, tz);
        Eigen::Matrix3d R = q.toRotationMatrix();
        Eigen::Matrix4d SE3Pose = Eigen::MatrixXd::Identity(4,4);
        SE3Pose.block(0,0,3,3) = R;
        SE3Pose.block(0,3,3,1) = t;

        trans.push_back(SE3Pose);
        loopPairs.push_back(res_pair);
        loopScores.push_back(score);
    }

    file.close();
}

void loadGlobalLoopPairs(std::string file_name,
                         std::vector<std::pair<unsigned int , unsigned int>>& loopPairs)
{
    loopPairs.clear();
    std::ifstream file(file_name, std::ios::in);
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << file_name << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        // std::cout<<line<<std::endl;
        int kf_index_1;
        int kf_index_2;
        iss >> kf_index_1 >> kf_index_2;
        // std::cout<<tx<<","<<ty<<","<<tz<<","<<
        //            qx<<","<<qy<<","<<qz<<","<<qw<<std::endl;
        std::pair<unsigned int , unsigned int> res_pair;
        res_pair.first = kf_index_1;
        res_pair.second = kf_index_2;
        loopPairs.push_back(res_pair);
    }

    file.close();
}


bool isKF(std::string kf_rgb_name, const std::vector<std::string>& rgb_list)
{
    bool is_kf = false;
    for(int j = 0; j < rgb_list.size(); ++j){
        if(rgb_list[j] == kf_rgb_name){
            return true;
        }
    }
    return false;
}

}
#endif