#ifndef WARPING_H
#define WARPING_H
#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <Eigen/Dense>
#include <fstream>
using namespace Eigen;
using namespace cv;
using namespace std;
class warping {
private:

	static double findDegree(const MatrixXd &m1,double d);
public:
	static MatrixXd rotation(const MatrixXd &m, double angle,int xc,int yc);
	static MatrixXd translation(const MatrixXd &m, int a,int b);
	static MatrixXd euclidien(const MatrixXd &m, double angle,int a,int b,int xc,int yc);
	static Mat warpingFinger(const Mat &img1, const Mat &img2);
};
#endif
