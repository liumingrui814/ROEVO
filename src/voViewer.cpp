#include "voViewer.h"

pangolin::OpenGlMatrix Eigen2gl(Eigen::Matrix4f matrix)
{
    pangolin::OpenGlMatrix glMatrix;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            glMatrix.m[i*4 + j] = matrix(j, i); // 注意：按列主序存储
        }
    }
    return glMatrix;
}

voViewer::voViewer(std::string windowName)
{
    windowName_ = windowName;
    pangolin::CreateWindowAndBind(windowName, 640,480);
    glEnable(GL_DEPTH_TEST);
    //3D visualizing window
    Eigen::Matrix4f view_point;
    // view_point << 0.987055, -0.0356286, 0.156861, -0.130428, 
    //              -0.0681276, -0.975983, 0.206956, -0.528214, 
    //                0.145719, -0.215005, -0.965752, -2.64094, 
    //                 0, 0, 0, 1; 
    //-- icl的视角
    // view_point << 0.927458, 0.188934, -0.32289, 0.24962, 
    //                0.212732, -0.976329, 0.0396091, -0.171934, 
    //               -0.307765, -0.105475, -0.945667, -1.64501,
    //               0, 0, 0, 1; 
    view_point << 0.927458, 0.188934, -0.32289, 0.197486, 
                  0.212732, -0.976329, 0.0396091, -0.141781, 
                 -0.307765, -0.105475, -0.945667, -1.98199, 
                  0,0,0,1;

    s_cam_ = std::shared_ptr<pangolin::OpenGlRenderState>(
            new pangolin::OpenGlRenderState(pangolin::ProjectionMatrix(640,480,420,420,320,240,0.02,500),
                                            view_point));

    handler_ = std::make_shared<pangolin::Handler3D>(*s_cam_);
    d_cam_ = &pangolin::CreateDisplay().SetBounds(0.0, 1.0, 0.0, 1.0,-640.0f/480.0f).SetHandler(handler_.get());
    
    //control part image window
    pangolin::CreatePanel("menu").SetBounds(0.3, 1, 0.0, 0.20);
    follow = std::make_shared<pangolin::Var<bool>>("menu.Follow", true, true);
    show_covisibility = std::make_shared<pangolin::Var<bool>>("menu.Co-visibility", true, true);
    slide_bar = std::make_shared<pangolin::Var<double>>("menu.slider", 0.5, 0, 1);

    cameraPose = Eigen::MatrixXd::Identity(4,4);
    gtPose = Eigen::MatrixXd::Identity(4,4);

    pangolin::GetBoundWindow()->RemoveCurrent();
    start();
}

void voViewer::render_loop()
{
    pangolin::BindToContext(windowName_);
    glEnable(GL_DEPTH_TEST);

    while (!pangolin::ShouldQuit() && !stop_)
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        pangolin::OpenGlMatrix glMatrix = Eigen2gl(gtPose.cast<float>());

        if(stop_) break;
        if(*follow == true){
            s_cam_->Follow(glMatrix);
        }

        if(stop_) break;
        d_cam_->Activate(*s_cam_);
        glClearColor(1.0f,1.0f,1.0f,1.0f);

        //绘制原点标准坐标轴
        Eigen::Matrix4d E = Eigen::MatrixXd::Identity(4,4);
        drawCoordinate(E,0.2);

        //绘制相机位姿
        drawCamera(cameraPose, cv::Vec3b(0,100,255), 0.06);

        //绘制局部地图滑窗
        for(int i = 0; i < sliding_window.size(); ++i)
        {
            drawCamera(sliding_window[i], cv::Vec3b(200,50,50), 0.03);
        }

        if(!localMap_cloud.empty()){
            for(size_t i = 0; i < localMap_cloud.size(); ++i)
            {
                drawPointCloudColorSequencial(localMap_cloud[i], cv::Vec3b(255, 50, 50), 2);
            }
        }

        if(!environment_cloud.empty())
        {
            for(size_t i = 0; i < environment_cloud.size(); ++i)
            {
                drawPointCloudColor(environment_cloud[i], cv::Vec3b(150, 150, 150), 2);
            }
        }

        if(*show_covisibility == true)
        {
            drawPointCloudColorful(covisibility_cloud, covisibility_color, 2);
        }

        //绘制轨迹
        if(trajectory.size() > 0){
            drawTrajectory(trajectory, false, cv::Vec3b(0,100,255), 0.05);
            drawTrajectory(trajectory, true, cv::Vec3b(0,100,255), 0.02);
        }

        //绘制真值轨迹
        if(trajectory_GT.size() > 0){
            drawTrajectory(trajectory_GT, false, cv::Vec3b(255,100,0), 0.05);
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(20));

        const Eigen::Matrix4f& modelViewMatrix = s_cam_->GetModelViewMatrix();
        // std::cout<<"*******************************************************"<<std::endl;
        // for (int i = 0; i < 4; ++i) {
        //     for (int j = 0; j < 4; ++j) {
        //         std::cout << modelViewMatrix(i, j) << " ";
        //     }
        //     std::cout << std::endl;
        // }


        pangolin::FinishFrame();
    }
}

void voViewer::start()
{
    stop_  = false;
    thread = std::thread(std::bind(&voViewer::render_loop, this));
    thread.detach();
}