
#ifndef FILTER_H
#define FILTER_H
#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <Eigen/Dense>
#include <fstream>
class filter{
private:
public:
    /**
     * calculate the 2d convolution
     */
	static Eigen::MatrixXd convolution(const Eigen::MatrixXd &kernel,const Eigen::MatrixXd &m);
    /**
     * calculate the 2d convolution using fft
     * @param I The image
     * @param k The kernel
     */
	static cv::Mat fft_matrix(const cv::Mat I, const Eigen::MatrixXd k);
    /**
     * generate a gaussian blur kernel
     * @param size The size of the kernel
     * @param sd The standard deviation of the gaussian blur
     */
	static Eigen::MatrixXd gaussianBlur(int size, double sd);
    /**
     * return the kernel of pixel at the coordinate (i,j) of the increasing gaussian blur
     * @param size The size of the kernel
     * @param m The image
     */
	static Eigen::MatrixXd kernelFunction(int i, int j, int size, const Eigen::MatrixXd &m);
    /**
     * perform an increasing gaussian blur
     * @param m The image
     * @param size_kernel The size of the gaussian blur kernel
     */
	static Eigen::MatrixXd convolution_local(const Eigen::MatrixXd &m,int size_kernel);
};
#endif
