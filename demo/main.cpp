#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <Eigen/Dense>
#include <fstream>
#include "util.h"
#include "warping.h"
#include "weak_finger.h"
#include <filter.h>
using namespace Eigen;
using namespace cv;
using namespace std;
int main(int argc, char* argv[]) {
	//Mat image = imread("resource/clean_finger.png",CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat image = imread(argv[1],CV_LOAD_IMAGE_GRAYSCALE);
	//Mat image2 = imread("weak_finger.png",CV_LOAD_IMAGE_GRAYSCALE);
	//Mat image3 = imread("resource/warp1_finger.png",CV_LOAD_IMAGE_GRAYSCALE);
	//cv::Mat image3 = imread(argv[2],CV_LOAD_IMAGE_GRAYSCALE);

	Eigen::MatrixXd m1 = util::getMatrix(image);
	//MatrixXd m2 = util::getMatrix(image3);


	double *a1 = new double(0);
	double *b1 = new double(0);
	int *xc1 = new int(0);
	int *yc1 = new int(0);
	//cout<<"degree = "<<degree<<endl;
	util::getInfo(m1,a1,b1,xc1,yc1);
	//double degree = findDegree(m2,1,*xc1,*yc1);
	int xc = 3*m1.rows()/4;
	int yc = m1.cols()/2;
	Eigen::MatrixXd m4 = warping::squeezedRotation(m1,-30,xc,yc);
	//Eigen::MatrixXd m4 = warping::rotation(m1,-45,*xc1,*yc1);
	//cout<<"haha"<<endl;
	//Mat im3 = warping::warpingFinger(image,image3);
	cv::Mat im3 = util::getImage(m4);
	cout<<"right"<<endl;
	Matrix3d m,h;
	m << 1,2,3,
		 4,5,6,
		 7,8,9;
	h << -1,-2,-1,
		  0,0,0,
		  1,2,1;

	MatrixXd mc = filter::convolution(h,m);
	cout<<"m = "<<mc<<endl;
  	namedWindow( "Display Image", WINDOW_AUTOSIZE );
  	imshow( "Display Image", im3);
  	imwrite("./warp1.png",im3);
  	waitKey(0);
  	return 0;
}
