# Specify the minimum version for CMake
cmake_minimum_required(VERSION 3.1.0)

project(Image CXX)

set(CMAKE_CXX_FLAGS "-std=c++11 -g -Wall -Wextra")
# Requiring OpenCV
find_package(OpenCV REQUIRED)

# Requiring Eigen
find_package(Eigen3 REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include ${EIGEN3_INCLUDE_DIR})

add_subdirectory(src)
add_subdirectory(demo)
