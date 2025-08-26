#ifndef TUM_FILE_H
#define TUM_FILE_H
#include <iomanip> // 注意包含这个头文件, 有效数字需要
#include <iostream>
#include <fstream>
#include <vector>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
namespace tum_file{

//-- 数一个文件有多少行(实际上是多少个/n换行符)
int CountLines(std::string filename){
    std::ifstream ReadFile;
    int n=0;
    std::string tmp;
    ReadFile.open(filename,std::ios::in);
    if(!ReadFile.is_open()){
        return -1;
    }else{
        while(std::getline(ReadFile,tmp,'\n'))n++;
        ReadFile.close();
        return n;
    }
}

/**
 * @brief 根据tum数据集目录下的associates文件获得关联好的rgb图与深度图的时间戳与文件名列表
 * @details associates.txt的每行满足 stamp rgb_name stamp depth_name
 * @param tum_path associates文件夹所在的目录，以"/"结尾
 * @param rgb_file_seq 彩色图图像文件名序列
 * @param depth_file_seq 深度图图像文件名序列
 * @param rgb_stamp_seq rgb时间戳序列
 * @param depth_stamp_seq depth时间戳序列
*/
void getTUMsequence(std::string tum_path,    
                    std::vector<std::string>& rgb_file_seq,
                    std::vector<std::string>& depth_file_seq,
                    std::vector<double>& rgb_stamp_seq,
                    std::vector<double>& depth_stamp_seq)
{
    //-- 清空文件列表
    rgb_file_seq.clear();
    depth_file_seq.clear();
    rgb_stamp_seq.clear();
    depth_stamp_seq.clear();
    //-- 得到文件关联的txt
    std::string associate = tum_path + "associates.txt";
    std::ifstream file(associate);
    std::string line;
    //-- 循环读取每行，每行的内容即为line
    while(std::getline(file, line)){
        std::istringstream iss(line);
        std::string word;
        //-- 处理一行中的所有内容，TUM格式为 rgb_stamp, rgb_file_name, depth_stamp, depth_file_name
        //-- rgb_stamp
        iss>>word;
        double time_stamp_rgb = std::stod(word);
        rgb_stamp_seq.push_back(time_stamp_rgb);
        //-- rgb_file_name
        iss>>word;
        std::string rgb_file_name = word;
        rgb_file_seq.push_back(rgb_file_name);
        //-- depth_stamp
        iss>>word;
        double time_stamp_depth = std::stod(word);
        depth_stamp_seq.push_back(time_stamp_depth);
        //-- depth_file_name
        iss>>word;
        std::string depth_file_name = word;
        depth_file_seq.push_back(depth_file_name);
    }
    file.close();
}

/**
 * @brief 读取groundtruth.txt文件得到每个时间戳对应的位姿真值
 * @details 给出的gt_path要直接是groundtruth.txt的实际目录
 *          tum的数据集格式：timestamp tx ty tz qx qy qz qw
 * @param gt_path 真值文件groundtruth.txt
 * @param TransVector 位姿列表
 * @param timeVector 时间戳列表
 */
void getGTsequence(std::string gt_path,    
                   std::vector<Eigen::Matrix4d>& TransVector,
                   std::vector<double>& timeVector)
{
    //-- 清空文件列表
    TransVector.clear();
    timeVector.clear();
    int poseLength = CountLines(gt_path);
    std::ifstream ifs(gt_path, std::ios::in);
    double tx, ty, tz, qx, qy, qz, qw;
    double timeStamp;
    if (ifs.is_open()){
        for (int i = 0; i < poseLength; i++){
            ifs>>timeStamp>>tx>>ty>>tz>>qx>>qy>>qz>>qw;
            Eigen::Quaterniond q(qw, qx, qy, qz);
            Eigen::Vector3d t(tx, ty, tz);
            Eigen::Matrix3d R = q.toRotationMatrix();
            Eigen::Matrix4d SE3Pose = Eigen::MatrixXd::Identity(4,4);
            SE3Pose.block(0,0,3,3) = R;
            SE3Pose.block(0,3,3,1) = t;
            TransVector.push_back(SE3Pose);
            timeVector.push_back(timeStamp);
        }
        ifs.close();
    }
    return;
}

//-- 更新一个时间戳+位姿
void updateStampedposeFile(Eigen::Matrix4d Trans, double timeStamp, std::string filename)
{
    std::ofstream fout;
    fout.open(filename,std::ios::app);
    Eigen::Vector3d translation = Trans.block(0,3,3,1);
    Eigen::Matrix3d rotation = Trans.block(0,0,3,3);
    Eigen::Quaterniond q(rotation);
    if(fout.is_open()){
        fout << std::setprecision(20);
        fout << timeStamp  <<" ";
        fout << std::setprecision(6);
        fout << translation(0) <<" "
             << translation(1) <<" "
             << translation(2) <<" "
            << q.x() <<" "
            << q.y() <<" "
            << q.z() <<" "
            << q.w() <<std::endl;
        fout.close();
    }else{
        std::cout<<"can't open file"<<std::endl;
    }
}

//-- 将一组时间戳与对应的位姿保存成TUM格式，存储在folder目录的poses.txt下
void saveEvaluationFiles(std::string filename, std::vector<Eigen::Matrix4d> PoseVector, std::vector<double> stamps)
{
    //-- 判断filename是不是.txt结尾的
    std::string suffix = ".txt";
    bool endWithtxt = true;
    if (filename.length() >= suffix.length()){
        endWithtxt =  (0 == filename.compare(filename.length() - suffix.length(), suffix.length(), suffix));
    }else{
        endWithtxt = false;
    }
    if(endWithtxt == false){
        std::cout<<"file suffix not txt, interrupt!"<<std::endl;
        return;
    }
    //-- 现在默认结尾是txt结尾的
    std::string poseFileName = filename;
    std::ofstream pose_writer(poseFileName, std::ios::out);
    for(std::size_t i = 0; i<PoseVector.size(); ++i)
    {
        Eigen::Matrix4d Trans = PoseVector[i];
        updateStampedposeFile(Trans, stamps[i], poseFileName);
    }
}

/**
 * @brief 保存一对真值轨迹值与代码轨迹值的pair, 用于evo的轨迹评测
 * @details 代码产生的时间戳与对应的位姿保存成TUM格式，存储在folder目录的poses.txt下
 *          时间戳对应插值得到的位姿真值同样保存成TUM格式存储在folder目录的gt.txt下
 * @param folder 文件存储目录，推荐rgb与depth所在的目录
 * @param PoseVector 自己的轨迹
 * @param stamps 时间戳，真值的时间戳与自己轨迹的时间戳要一样
 * @param gtVector 真值的轨迹，对应索引的时间戳与自己的轨迹对应索引一致
*/
void saveEvaluationPair(std::string folder, 
                        std::vector<Eigen::Matrix4d> PoseVector, 
                        std::vector<double> stamps,
                        std::vector<Eigen::Matrix4d> gtVector)
{
    std::string poseFileName = folder + "poses.txt";
    std::string gtFileName = folder + "gt.txt";
    //-- 检查四个列表大小是否相同
    bool isSame = (PoseVector.size()==stamps.size()) && 
                  (PoseVector.size()==gtVector.size());
    if(!isSame){
        std::cout<<"\033[31m [ERROR] \033[0m"<<": evo pairs not match!"<<std::endl;
        return;
    }
    std::ofstream pose_writer(poseFileName, std::ios::out);
    for(std::size_t i = 0; i<PoseVector.size(); ++i)
    {
        Eigen::Matrix4d Trans = PoseVector[i];
        updateStampedposeFile(Trans, stamps[i], poseFileName);
    }
    std::ofstream gt_writer(gtFileName, std::ios::out);
    for(std::size_t i = 0; i < gtVector.size(); ++i)
    {
        Eigen::Matrix4d Trans = gtVector[i];
        updateStampedposeFile(Trans, stamps[i], gtFileName);
    }
}


}

#endif