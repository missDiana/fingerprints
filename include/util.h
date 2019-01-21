#ifndef UTIL_H
#define UTIL_H
#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <Eigen/Dense>
#include <fstream>
using namespace Eigen;
using namespace cv;
using namespace std;

class util {
private:
	static double fx(int i,int j,MatrixXd &m);
	static double fy(int i,int j,MatrixXd &m);
	static double fxy(int i,int j,MatrixXd &m);
	static double power(double a, double b);
public:
	static double distance(int x1, int y1, int x2, int y2);
	static MatrixXd getMatrix(const Mat &img);
	static Mat getImage(const MatrixXd &m);
	static double getMajorAxis(const MatrixXd &image);
	static double getMinorAxis(const MatrixXd &image);
	static MatrixXd bilinear_inter(const MatrixXd &m);
	static void bicubic_inter(MatrixXd &m);
	static void getInfo(const MatrixXd &m,double *axisL,double *axisS,int *xc,int *yc);
	static void range(Mat img);
	static Mat changeBlock(Mat &image);
	static Mat sym_x(Mat &image);
	static Mat sym_y(Mat &image);
	static Mat sym_x_diag(Mat &image);
	static Mat sym_y_diag(Mat &image);
	static void getCenter(const MatrixXd &m,int *x, int *y);

};
#endif
