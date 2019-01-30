
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
	static Eigen::MatrixXd convolution(const Eigen::MatrixXd &kernel,const Eigen::MatrixXd &m);
	static cv::Mat fft_matrix(const cv::Mat I, const Eigen::MatrixXd k);
	static Eigen::MatrixXd gaussianBlur(int size, double sd);
	static Eigen::MatrixXd kernelFunction(int i, int j, int size, const Eigen::MatrixXd &m);
	static Eigen::MatrixXd convolution_local(const Eigen::MatrixXd &m,int size_kernel);
};
#endif
