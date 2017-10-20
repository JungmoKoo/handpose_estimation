cmake_minimum_required(VERSION 2.8.3)
project(handpose_estimation)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(ROS_BUILD_TYPE Release)

# please edit path
SET(CAFFE_HOME /home/jungmo/mylibs/caffe)
set(BOOST_LIB boost_system boost_filesystem boost_thread)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  cv_bridge
  image_transport
)
find_package(CUDA REQUIRED)
find_package(Caffe REQUIRED)
find_package(Boost REQUIRED)

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES handpose_estimation
#  CATKIN_DEPENDS roscpp rospy std_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########

include_directories(
  /usr/local/cuda/include
  ${CAFFE_HOME}/build/include
  include
  ${catkin_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME}_node src/main.cpp include/fcn/fcn_utils.cpp)

target_link_libraries(${PROJECT_NAME}_node
   ${catkin_LIBRARIES}
   ${BOOST_LIB}
   caffe
   glog gflags protobuf m hdf5_hl hdf5
   cuda
   cblas atlas
   stdc++
)
