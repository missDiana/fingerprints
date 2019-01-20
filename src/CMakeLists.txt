# Specify the minimum version for CMake

cmake_minimum_required(VERSION 2.8)
project(Image)
find_package(OpenCV REQUIRED)
add_executable(Image util.cpp)
include_directories(/usr/lib/x86_64-linux-gnu/)
target_link_libraries(util libopencv_calib3d.so)