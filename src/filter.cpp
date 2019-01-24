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
