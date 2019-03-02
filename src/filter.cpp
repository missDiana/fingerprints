
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

cv::Mat filter::fft_matrix(const cv::Mat I, const Eigen::MatrixXd k) {
	Mat kernel(k.rows(),k.cols(),CV_32F);
	for(int i=0;i<k.rows();i++){
		for(int j=0;j<k.cols();j++){
			kernel.at<float>(i,j,0) = k(i,j);
		}
	}
	cv::Mat kernel_padded;
	int N1 = I.rows;
	int N2 = I.cols;
	cv::copyMakeBorder(kernel, kernel_padded, 0, N1 - kernel.rows, 0, N2 - kernel.cols, BORDER_CONSTANT, Scalar::all(0));
	cv::Mat planesI[] = {Mat_<float>(I), Mat::zeros(I.size(), CV_32F)};
    cv::Mat complexI;
    cv::merge(planesI, 2, complexI);
	cv::Mat planesKernel[] = {Mat_<float>(kernel_padded), Mat::zeros(kernel_padded.size(), CV_32F)};
    cv::Mat complexKernel;
    cv::merge(planesKernel, 2, complexKernel);

	cv::dft(complexI, complexI);
	cv::dft(complexKernel, complexKernel);

	cv::split(complexI, planesI);
	cv::split(complexKernel, planesKernel);
	cv::Mat resultReal(planesI[0].rows,planesI[0].cols,CV_32F);
	cv::Mat resultImag(planesI[1].rows,planesI[1].cols,CV_32F);

	for(int i=0;i<resultReal.rows;i++) {
		for(int j=0;j<resultReal.cols;j++) {
			resultReal.at<float>(i,j,0) = planesI[0].at<float>(i,j,0)*planesKernel[0].at<float>(i,j,0)-planesI[1].at<float>(i,j,0)*planesKernel[1].at<float>(i,j,0);
			resultImag.at<float>(i,j,0) = planesI[0].at<float>(i,j,0)*planesKernel[1].at<float>(i,j,0)+planesI[1].at<float>(i,j,0)*planesKernel[0].at<float>(i,j,0);
		}
	}
	cv::Mat resultplane[] = {Mat_<float>(resultReal), Mat_<float>(resultImag)};
	cv::Mat resultComplex;
	cv::merge(resultplane, 2, resultComplex);
	cv::dft(resultComplex, resultComplex,cv::DFT_INVERSE + cv::DFT_SCALE);
	cv::split(resultComplex, resultplane);
	cv::magnitude(resultplane[0], resultplane[1], resultplane[0]);// planes[0] = magnitude
    Mat magI = resultplane[0];
	Mat img(magI.rows,magI.cols,CV_8UC1);
	for(int i=0;i<img.rows;i++){
		for(int j=0;j<img.cols;j++){
			int val = round(magI.at<float>(i,j,0));
			if(val>255) val = 255;
			if(val<0) val = 0;
			img.at<uchar>(i,j,0) = (uchar)val;
		}
	}
	return img;
}

Eigen::MatrixXd filter::gaussianBlur(int size, double sd) {
	MatrixXd g = MatrixXd::Zero(size,size);
	int index = size/2;
	if (sd==0) {
		g(size/2,size/2) = 1;
	}
	else {
		for(int i=0;i<size;i++) {
			for(int j=0;j<size;j++){
				double x = i - index;
				double y = j - index;
				g(i,j) = 1/(2*M_PI*sd*sd)*exp(-(x*x+y*y)/(2*sd*sd));
			}
		}
	}
	return g/g.sum();
}

Eigen::MatrixXd filter::kernelFunction(int i, int j, int size, const Eigen::MatrixXd &m) {
	double k = 1;
	int *xc = new int(0);
	int *yc = new int(0);
	util::getCenter(m,xc,yc);
	return gaussianBlur(size,util::distance(i,j,*xc,*yc)*k);
}

Eigen::MatrixXd filter::convolution_local(const Eigen::MatrixXd &m,int size_kernel) {
	MatrixXd mc = m;
	int k = (size_kernel-1)/2;
	for (int i=0;i<m.rows();i++) {
		for (int j=0;j<m.cols();j++) {
			mc(i,j) = 1;
		}
	}
	int *xc = new int(0);
	int *yc = new int(0);
	util::getCenter(m,xc,yc);
	Vector4d d;
	d << util::distance(0,0,*xc,*yc),util::distance(m.rows(),0,*xc,*yc),util::distance(0,m.cols(),*xc,*yc),util::distance(m.rows(),m.cols(),*xc,*yc);
	MatrixXf::Index max;
  	double dmax = d.maxCoeff(&max);
	cout<<"dmax = "<<dmax<<endl;
	int c = 5;
	int num = 1+(int)(dmax/c);
	MatrixXd *ker = new MatrixXd[num];
	for (int i=0;i<num;i++) {
		ker[i] = gaussianBlur(size_kernel,1.5*log(i/6.0+1));
	}
	int n = 0;
	for (int i=k;i<m.rows()-k;i++) {
		for (int j=k;j<m.cols()-k;j++) {
			double sum = 0;
			int val = (int)(util::distance(i,j,*xc,*yc)/c);
			MatrixXd kernel = ker[val];
			for(int x=0;x<size_kernel;x++) {
				for(int y=0;y<size_kernel;y++){
					sum = sum + kernel(kernel.rows()-1-x,kernel.cols()-1-y)*m(i-k+x,j-k+y);
				}
			}
			mc(i,j) = sum;
			n++;
			cout<<n<<"/"<<pow((m.rows()-2*k),2)<<" done"<<endl;
		}
	}
	return mc;
}
