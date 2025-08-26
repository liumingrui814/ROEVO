cmake_minimum_required(VERSION 2.8)
project(vo1)

set(CMAKE_BUILD_TYPE "Release")
add_definitions("-DENABLE_SSE")
# set(CMAKE_CXX_FLAGS "-std=c++14 -O2 ${SSE_FLAGS} -g -march=native")
set(CMAKE_CXX_FLAGS "-std=c++14 -O2 ${SSE_FLAGS} -g")

find_package(OpenCV 3.4 REQUIRED)
find_package(PCL REQUIRED)
find_package(Sophus REQUIRED)
find_package(FMT REQUIRED)
# find_package(Ceres REQUIRED)
find_package(Pangolin REQUIRED)
find_package(TBB REQUIRED)

include_directories(
  include
  utils
    ${catkin_INCLUDE_DIRS} 
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${Sophus_INCLUDE_DIRS}
    # ${CERES_INCLUDE_DIRS}
    # ${GTSAM_INCLUDE_DIR}
    ${Pangolin_INCLUDE_DIRS}
    ${TBB_INCLUDE_DIRS}
)
include_directories("/usr/include/eigen3")


# 查找并链接 libtbb.so.2
find_library(TBB_LIB
    NAMES tbb libtbb.so.2
    PATHS /usr/lib/x86_64-linux-gnu
)


add_library(Edge
    include/edge.h
    src/edge.cpp
)
target_link_libraries(Edge ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})

add_library(EdgeCluster
    include/edgeCluster.h
    src/edgeCluster.cpp
)
target_link_libraries(EdgeCluster ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})

add_library(edgeSelector
    include/edgeSelector.h
    src/edgeSelector.cpp
)
target_link_libraries(edgeSelector ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} Edge EdgeCluster TBB::tbb)

add_library(Frame
    include/Frame.h
    src/Frame.cpp
)
target_link_libraries(Frame ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} edgeSelector)

add_library(KeyFrame
    include/KeyFrame.h
    src/KeyFrame.cpp
)
target_link_libraries(KeyFrame ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} fmt::fmt)

add_library(elementEdge
    include/elementEdge.h
    src/elementEdge.cpp
)
target_link_libraries(elementEdge ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} fmt::fmt)

add_library(featureMerger
    include/featureMerger.h
    src/featureMerger.cpp
)
target_link_libraries(featureMerger ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${PCL_LIBRARIES} KeyFrame elementEdge fmt::fmt)

add_library(localMap
    include/localMap.h
    src/localMap.cpp
)
target_link_libraries(localMap ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${PCL_LIBRARIES} KeyFrame elementEdge featureMerger fmt::fmt)

add_library(weight
    include/robustWeight.h
    src/robustWeight.cpp
)
target_link_libraries(weight ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} fmt::fmt)

# add_library(geometry
#     include/geometryFactor.h
#     src/geometryFactor.cpp
# )
# target_link_libraries(geometry ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} fmt::fmt)

# add_library(photometric
#     include/photometricFactor.h
#     src/photometricFactor.cpp
# )
# target_link_libraries(photometric ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} fmt::fmt)

add_library(directTracker
    include/directTracker.h
    src/directTracker.cpp
)
target_link_libraries(directTracker ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} weight fmt::fmt TBB::tbb)

add_library(fineTracker
    include/fineTracker.h
    src/fineTracker.cpp
)
target_link_libraries(fineTracker ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} Frame weight fmt::fmt TBB::tbb)

add_library(Optimizer
    include/Optimizer.h
    src/Optimizer.cpp
)
target_link_libraries(Optimizer ${catkin_LIBRARIES} 
        ${OpenCV_LIBRARIES} 
        ${PCL_LIBRARIES} 
        KeyFrame elementEdge fineTracker featureMerger fmt::fmt TBB::tbb)

add_library(paramLoader
    include/paramLoader.h
    src/paramLoader.cpp
)
target_link_libraries(paramLoader ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})

add_library(vizBase
    utils/visualizerBase.h
    utils/visualizerBase.cpp
)
target_link_libraries(vizBase ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${Pangolin_LIBRARIES})

add_library(voViewer
    include/voViewer.h
    src/voViewer.cpp
)
target_link_libraries(voViewer ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${Pangolin_LIBRARIES} vizBase)

#####################################################################
#-------------------------------------------------------------------#
# executable files for system usage                                 #
#-------------------------------------------------------------------#
#####################################################################

add_executable(coarse examples/coarseTracking.cpp)
target_link_libraries(coarse ${OpenCV_LIBS} ${PCL_LIBRARIES} edgeSelector Frame directTracker paramLoader fmt::fmt)

add_executable(oedge examples/testOrganizedEdges.cpp)
target_link_libraries(oedge ${OpenCV_LIBS} ${PCL_LIBRARIES} Frame edgeSelector fmt::fmt paramLoader)

add_executable(fine examples/fineTracking.cpp)
target_link_libraries(fine ${OpenCV_LIBS} ${PCL_LIBRARIES} Frame fineTracker fmt::fmt paramLoader)

add_executable(lmap examples/testLocalMap.cpp)
target_link_libraries(lmap ${OpenCV_LIBS} ${PCL_LIBRARIES} edgeSelector Frame KeyFrame localMap fmt::fmt paramLoader)

add_executable(ba examples/testBA.cpp)
target_link_libraries(ba ${OpenCV_LIBS} ${PCL_LIBRARIES} edgeSelector Frame KeyFrame localMap fmt::fmt paramLoader Optimizer)

add_executable(win examples/slidingWindow.cpp)
target_link_libraries(win ${OpenCV_LIBS} ${PCL_LIBRARIES} edgeSelector Frame KeyFrame localMap fmt::fmt paramLoader voViewer)

add_executable(winBA examples/localMapping.cpp)
target_link_libraries(winBA ${OpenCV_LIBS} ${PCL_LIBRARIES} 
                      edgeSelector Frame KeyFrame localMap fmt::fmt paramLoader voViewer Optimizer)









