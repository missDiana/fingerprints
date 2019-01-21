#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <Eigen/Dense>
#include <fstream>
#include "util.h"
#include "warping.h"
#include "weak_finger.h"
using namespace Eigen;
using namespace cv;
using namespace std;
int main() {
	Mat image = imread("resource/clean_finger.png",CV_LOAD_IMAGE_GRAYSCALE);
	//Mat image2 = imread("weak_finger.png",CV_LOAD_IMAGE_GRAYSCALE);
	Mat image3 = imread("resource/warp1_finger.png",CV_LOAD_IMAGE_GRAYSCALE);
	MatrixXd m1 = util::getMatrix(image);
	MatrixXd m2 = util::getMatrix(image3);
	double *a1 = new double(0);
	double *b1 = new double(0);
 	int *xc1 = new int(0);
	int *yc1 = new int(0);
	//cout<<"degree = "<<degree<<endl;
	util::getInfo(m2,a1,b1,xc1,yc1);
	//double degree = findDegree(m2,1,*xc1,*yc1);
	MatrixXd m4 = warping::rotation(m1,45,*xc1,*yc1);
	Mat im3 = util::getImage(m4);
  	namedWindow( "Display Image", WINDOW_AUTOSIZE );
  	imshow( "Display Image", im3);
  	imwrite("./warp1.png",im3);
  	waitKey(0);
  	return 0;
}
