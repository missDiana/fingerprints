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
	/**
	symmetric schemes for first order derivative on direction x
	*/
	static double fx(int i,int j,MatrixXd &m);

	/**
	symmetric schemes for first order derivative on direction y
	*/
	static double fy(int i,int j,MatrixXd &m);

	/**
	symmetric schemes for second order derivative
	*/
	static double fxy(int i,int j,MatrixXd &m);

	/**
	if b>=0 return a^b, else return 0
	*/
	static double power(double a, double b);
public:
	/**
	The distance between two pixels
	*/
	static double distance(int x1, int y1, int x2, int y2);

	/**
	convert Mat image file into matrix
	*/
	static MatrixXd getMatrix(const Mat &img);

	/**
	convert matrix into Mat image file
	*/
	static Mat getImage(const MatrixXd &m);

	/**
	Return the major axis of the fingerprint
	*/
	static double getMajorAxis(const MatrixXd &image);

	/**
	Return the minor axis of the fingerprint
	*/
	static double getMinorAxis(const MatrixXd &image);

	/**
	The bilinear interpolation. Pixels need interpolating should = -1
	*/
	static MatrixXd bilinear_inter(const MatrixXd &m);

	/**
	The bicubic interpolation. Pixels need interpolating should = -1
	*/
	static void bicubic_inter(MatrixXd &m);

	/**
	The values of major axis, minor axis, coordinate of the center are resp. restored
	in axisL, axisS, xc and yc
	*/
	static void getInfo(const MatrixXd &m,double *axisL,double *axisS,int *xc,int *yc);

	/**
	Print the max and min intensity in a grey scale image
	*/
	static void range(Mat img);

	/**
	change some blocks of an image into black and white
	*/
	static Mat changeBlock(Mat &image);

	/**
	Symmetric transform along the axis x
	*/
	static Mat sym_x(Mat &image);

	/**
	Symmetric transform along the axis y
	*/
	static Mat sym_y(Mat &image);

	/**
	Symmetric transform along the diagonal x
	*/
	static Mat sym_x_diag(Mat &image);

	/**
	Symmetric transform along the diagonal y
	*/
	static Mat sym_y_diag(Mat &image);

	/**
	Restore the center of the fingerprint into x and y
	*/
	static void getCenter(const MatrixXd &m,int *x, int *y);

};
#endif
