
#include <util.h>
#include <warping.h>
#include <weak_finger.h>
#include <filter.h>
#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <Eigen/Dense>
#include <fstream>
Eigen::MatrixXd filter::convolution(const Eigen::MatrixXd &kernel,const Eigen::MatrixXd &m) {
	MatrixXd mc = m;
	int k = (kernel.rows()-1)/2;
	for (int i=0;i<m.rows();i++) {
		for (int j=0;j<m.cols();j++) {
			mc(i,j) = 1;
		}
	}
	for (int i=k;i<m.rows()-k;i++) {
		for (int j=k;j<m.cols()-k;j++) {
			double sum = 0;
			for(int x=0;x<kernel.rows();x++) {
				for(int y=0;y<kernel.cols();y++){
					sum = sum + kernel(kernel.rows()-1-x,kernel.cols()-1-y)*m(i-k+x,j-k+y);
				}
			}
			mc(i,j) = sum;
		}
	}
	return mc;
}

cv::Mat filter::fft_matrix(const cv::Mat I, const cv::Mat kernel) {
	int N1 = I.rows + kernel.rows - 1;
	int N2 = I.cols + kernel.cols - 1;
	cv::Mat I_padded,kernel_padded;             //expand input image to optimal size
	cv::copyMakeBorder(I, I_padded, 0, N1 - I.rows, 0, N2 - I.cols, BORDER_CONSTANT, Scalar::all(0));
	cv::copyMakeBorder(kernel, kernel_padded, 0, N1 - kernel.rows, 0, N2 - kernel.cols, BORDER_CONSTANT, Scalar::all(0));

	cv::Mat I_efficient,kernel_efficient;
    int m = cv::getOptimalDFTSize(N1);
    int n = cv::getOptimalDFTSize(N2);
    cv::copyMakeBorder(I_padded,I_efficient,0,m-I_padded.rows,0,n - I_padded.cols,BORDER_CONSTANT,Scalar::all(0));
	cv::copyMakeBorder(kernel_padded,kernel_efficient,0,m-kernel_padded.rows,0,n - kernel_padded.cols,BORDER_CONSTANT,Scalar::all(0));

	cv::Mat planesI[] = {Mat_<float>(I_padded), Mat::zeros(I_padded.size(), CV_32F)};
    cv::Mat complexI;
    cv::merge(planesI, 2, complexI);
	cv::Mat planesKernel[] = {Mat_<float>(kernel_padded), Mat::zeros(kernel_padded.size(), CV_32F)};
    cv::Mat complexKernel;
    cv::merge(planesKernel, 2, complexKernel);

	cv::dft(complexI, complexI);
	cv::dft(complexKernel, complexKernel);

	cv::split(complexI, planesI);
	cv::split(complexKernel, planesKernel);

	cv::Mat resultReal[] = {Mat_<float>(planesI[0]), Mat::zeros(planesI[0].size(), CV_32F)};
	cv::Mat resultImag[] = {Mat_<float>(planesI[1]), Mat::zeros(planesI[1].size(), CV_32F)};

	for(int i=0;i<m;i++) {
		for(int j=0;j<n;j++) {
			resultReal[i,j] = planesI[0][i,j]*planesKernel[0][i,j]-planesI[1][i,j]*planesKernel[1][i,j];
			resultImag[i,j] = planesI[0][i,j]*planesKernel[1][i,j]+planesI[1][i,j]*planesKernel[0][i,j];
		}
	}

	cv::merge(resultReal, 2, resultImag);
	cv::Mat result;
	cv::dft(complexI, result,cv::DFT_INVERSE);
	return result;
}
