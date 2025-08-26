#include "visualizerBase.h"

VisualizerBase::VisualizerBase()
{
    stop_ = false;
    ColorTabel_JET =  generateColorMap(cv::COLORMAP_JET);
    ColorTabel_RAINBOW =  generateColorMap(cv::COLORMAP_RAINBOW);
    ColorTabel_PARULA =  generateColorMap(cv::COLORMAP_PARULA);
}

VisualizerBase::~VisualizerBase()
{
    if(!stop_) stop();
}

void VisualizerBase::drawPoint(cv::Point3d v1, cv::Vec3b color, int size){
    if(stop_ == true) return;
    glPointSize(size);
    glBegin(GL_POINTS);
        addPoint(v1, color);
    glEnd();
}

void VisualizerBase::drawLine(cv::Point3d v1, cv::Point3d v2, cv::Vec3b color, int width){
    if(stop_ == true) return;
    glLineWidth(width);
    glBegin ( GL_LINES );
        addLine(v1, v2, color);
    glEnd();
}

void VisualizerBase::drawCoordinate(const Eigen::Matrix4d& Transform, double length, int width){
    if(stop_ == true) return;
    Eigen::Vector3d origin = Transform.block(0,3,3,1);
    Eigen::Matrix3d axes = Transform.block(0,0,3,3);
    Eigen::Vector3d x_axis = axes.col(0)*length + origin;
    Eigen::Vector3d y_axis = axes.col(1)*length + origin;
    Eigen::Vector3d z_axis = axes.col(2)*length + origin;
    glLineWidth(width);         //indicate line width
    glBegin ( GL_LINES );       //start plotting a line
        addLine(cv::Point3d(origin(0),origin(1),origin(2)),cv::Point3d(x_axis(0),x_axis(1),x_axis(2)),cv::Vec3b(200,0,0));
        addLine(cv::Point3d(origin(0),origin(1),origin(2)),cv::Point3d(y_axis(0),y_axis(1),y_axis(2)),cv::Vec3b(0,200,0));
        addLine(cv::Point3d(origin(0),origin(1),origin(2)),cv::Point3d(z_axis(0),z_axis(1),z_axis(2)),cv::Vec3b(0,0,200));
    glEnd();
}

void VisualizerBase::drawTrajectory(const std::vector<Eigen::Matrix4d>& poses, bool showRotation,cv::Vec3b color, double axis_len)
{
    if(stop_ == true) return;
    if(poses.empty()) return;
    if(!showRotation){
        glLineWidth(3);         //indicate line width
        glBegin ( GL_LINES );   //start plotting a line
        for(size_t i = 0; i<poses.size()-1; ++i){
            addLine(cv::Point3d(poses[i](0,3),poses[i](1,3),poses[i](2,3)),
                    cv::Point3d(poses[i+1](0,3),poses[i+1](1,3),poses[i+1](2,3)),color);
        }
        glEnd();
    }else{
        for(auto& pose:poses){
            drawCoordinate(pose,axis_len,1);
        }
    }
}

void VisualizerBase::drawPointCloudColor(const std::vector<cv::Point3d>& pointcloud, cv::Vec3b color, int pointSize){
    if(stop_ == true) return;
    glPointSize(pointSize);
    glBegin(GL_POINTS);
    for(const auto& point: pointcloud){
        addPoint(point,color);
    }
    glEnd();
}

void VisualizerBase::drawPointCloudColorSequencial(const std::vector<cv::Point3d>& pointcloud, cv::Vec3b color, int pointSize)
{
    if(stop_ == true) return;
    glPointSize(pointSize);
    glBegin(GL_POINTS);
    for(const auto& point: pointcloud){
        addPoint(point,color);
    }
    glEnd();
    
    if(pointcloud.size() <= 1) return;

    glBegin(GL_LINES);
    for(size_t i = 1; i < pointcloud.size(); ++i){
        addLine(pointcloud[i-1], pointcloud[i], color);
    }
    glEnd();
}

void VisualizerBase::drawPointCloudAxis(const std::vector<cv::Point3d>& pointcloud, std::string axis, int pointSize){
    if(stop_ == true) return;
    if(axis != std::string("x") && axis != std::string("y") && axis != std::string("z")){
        std::cout<<"wrong input type! please input x,y,or z"<<std::endl;
    }
    //find maximum and minimum point
    float maxC = FLT_MIN;
    float minC = FLT_MAX;
    float currC;
    for(size_t i = 0; i<pointcloud.size(); ++ i){
        if(axis==std::string("x"))currC=pointcloud[i].x;
        if(axis==std::string("y"))currC=pointcloud[i].y;
        if(axis==std::string("z"))currC=pointcloud[i].z;
        if(currC>maxC)maxC=currC;
        if(currC<minC)minC=currC;
    }
    //assign color
    glPointSize(pointSize);
    glBegin(GL_POINTS);
    for(const auto& point: pointcloud){
        if(axis==std::string("x"))currC=point.x;
        if(axis==std::string("y"))currC=point.y;
        if(axis==std::string("z"))currC=point.z;
        int indec = int((currC-minC)/(maxC-minC)*255);
        addPoint(point,ColorTabel_PARULA.at<cv::Vec3b>(indec,0));
    }
    glEnd();
}

void VisualizerBase::drawPointCloudColorful(const std::vector<cv::Point3d>& pointcloud, 
                        const std::vector<cv::Vec3b>& colorMap,
                        int pointSize){
    if(stop_ == true) return;
    if(pointcloud.size() != colorMap.size()){
        std::cout<<"wrong input type! pointcloud and colorMap have to be the same size"<<std::endl;
    }
    glPointSize(pointSize);
    glBegin(GL_POINTS);
    for(size_t i = 0; i<pointcloud.size(); ++i){
        addPoint(pointcloud[i],colorMap[i]);
    }
    glEnd();
}

void VisualizerBase::drawCamera(Eigen::Matrix4d cameraPose, cv::Vec3b color, double camSize){
    if(stop_ == true) return;
    //画相机
    const float w=camSize;
    const float h=w*0.75;
    const float z=w*0.6;
    //使用位置变换矩阵
    glPushMatrix();
    Eigen::Matrix3d R_d = cameraPose.block(0,0,3,3);
    Eigen::Vector3d t_d = cameraPose.block(0,3,3,1);
    Eigen::Matrix3f R = R_d.cast<float>();
    Eigen::Vector3f t = t_d.cast<float>();
    //变换如该矩阵，注意这个变换矩阵是转置的
    std::vector<GLfloat> Twc ={ R(0,0), R(1,0), R(2,0), 0,
                                R(0,1), R(1,1), R(2,1), 0,
                                R(0,2), R(1,2), R(2,2), 0,
                                 t.x(),  t.y(),  t.z(), 1};
    //变换
    glMultMatrixf(Twc.data());
    glLineWidth(2);
    glBegin(GL_LINES);
        addLine(cv::Point3d(0,0,0), cv::Point3d(w,h,z), color);
        addLine(cv::Point3d(0,0,0), cv::Point3d(-w,-h,z), color);
        addLine(cv::Point3d(0,0,0), cv::Point3d(-w,h,z), color);
        addLine(cv::Point3d(0,0,0), cv::Point3d(w,-h,z), color);
        addLine(cv::Point3d(w,h,z), cv::Point3d(w,-h,z), color);
        addLine(cv::Point3d(-w,h,z), cv::Point3d(-w,-h,z), color);
        addLine(cv::Point3d(-w,h,z), cv::Point3d(w,h,z), color);
        addLine(cv::Point3d(-w,-h,z), cv::Point3d(w,-h,z), color);
    glEnd();
    glPopMatrix();
}

void VisualizerBase::draw_image(const cv::Mat& image, pangolin::View* port)
{
    glGetError();
    pangolin::GlTexture image_texture(image.cols, image.rows, GL_RGB, false, 0, GL_BGR,GL_UNSIGNED_BYTE);
    image_texture.Upload(image.data,GL_BGR,GL_UNSIGNED_BYTE);
    port->Activate();
    glColor3d(1.0,1.0,1.0);
    image_texture.RenderToViewportFlipY();
}

void VisualizerBase::start()
{
    stop_  = false;
    thread = std::thread(std::bind(&VisualizerBase::render_loop, this));
    thread.detach();
}

void VisualizerBase::stop()
{
    stop_ = true;

    for (int i = 0; i < 100; i++) { // 尝试 100 次，每次 10ms
        if (!thread.joinable()) break; // 线程已退出
        std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(10.0));
    }

}

void VisualizerBase::addPoint(cv::Point3d v1, cv::Vec3b color){
    glColor3d (float(color(0))/255.0,float(color(1))/255.0,float(color(2))/255.0);
    glVertex3d(v1.x,v1.y,v1.z);
}

void VisualizerBase::addLine(cv::Point3d v1, cv::Point3d v2, cv::Vec3b color){
    glColor3d (float(color(0))/255.0,float(color(1))/255.0,float(color(2))/255.0);
    glVertex3d(v1.x,v1.y,v1.z);
    glVertex3d(v2.x,v2.y,v2.z);
}

cv::Mat VisualizerBase::generateColorMap(cv::ColormapTypes mapType){
    cv::Mat valueTabel(256,1,CV_8UC1);
    cv::Mat ColorTabel;
    for(int i = 0; i<256; i++){
        valueTabel.at<uint8_t>(i,0)=i;
    }
    cv::applyColorMap(valueTabel,ColorTabel,mapType);
    cv::cvtColor(ColorTabel,ColorTabel,cv::COLOR_BGR2RGB);
    return ColorTabel;
}
