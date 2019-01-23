add_executable(main main.cpp)

target_link_libraries(main ${Eigen3_LIBS} ${OpenCV_LIBS} fingerprint)
