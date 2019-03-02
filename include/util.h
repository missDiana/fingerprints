#ifndef UTIL_H
#define UTIL_H

/// Use only includes needed in the header
#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <Eigen/Dense>
#include <fstream>

class util {
private:
	/**
	symmetric schemes for first order derivative on direction x
	*/
	static double fx(int i,int j,Eigen::MatrixXd &m);

	/**
	symmetric schemes for first order derivative on direction y
	*/
	static double fy(int i,int j,Eigen::MatrixXd &m);

	/**
	symmetric schemes for second order derivative
	*/
	static double fxy(int i,int j,Eigen::MatrixXd &m);

	/**
	if b>=0 return a^b, else return 0
	*/
	static double power(double a, double b);
public:
	/**
	The distance between two pixels
	*/
	static double distance(double x1, double y1, double x2, double y2);

	/**
	convert Mat image file into matrix
	*/
	static Eigen::MatrixXd getMatrix(const cv::Mat &img);

	/**
	convert matrix into Mat image file
	*/
	static cv::Mat getImage(const Eigen::MatrixXd &m);

	/**
	Return the major axis of the fingerprint
	*/
	static double getMajorAxis(const Eigen::MatrixXd &image);

	/**
	Return the minor axis of the fingerprint
	*/
	static double getMinorAxis(const Eigen::MatrixXd &image);

	/**
	The bilinear interpolation. Pixels need interpolating should = -1
	*/
	static Eigen::MatrixXd bilinear_inter(const Eigen::MatrixXd &m,const Eigen::MatrixXd &m_rotation,double theta,int xc,int yc);

	static Eigen::MatrixXd bilinear_inter_squeeze(const Eigen::MatrixXd &m,const Eigen::MatrixXd &m_rotation,double t,double ratio,int k,int xc,int yc);
	static Eigen::MatrixXd bilinear_inter_naive(const Eigen::MatrixXd &m);

	/**
	The bicubic interpolation. Pixels need interpolating should = -1
	*/
	static void bicubic_inter(Eigen::MatrixXd &m);

	/**
	The values of major axis, minor axis, coordinate of the center are resp. restored
	in axisL, axisS, xc and yc
	*/
	static void getInfo(const Eigen::MatrixXd &m,double *axisL,double *axisS,int *xc,int *yc);

	/**
	Print the max and min intensity in a grey scale image
	*/
	static void range(cv::Mat img);

	/**
	change some blocks of an image into black and white
	*/
	static cv::Mat changeBlock(cv::Mat &image);

	/**
	Symmetric transform along the axis x
	*/
	static cv::Mat sym_x(cv::Mat &image);

	/**
	Symmetric transform along the axis y
	*/
	static cv::Mat sym_y(cv::Mat &image);

	/**
	Symmetric transform along the diagonal x
	*/
	static cv::Mat sym_x_diag(cv::Mat &image);

	/**
	Symmetric transform along the diagonal y
	*/
	static cv::Mat sym_y_diag(cv::Mat &image);

	/**
	Restore the center of the fingerprint into x and y
	*/
	static void getCenter(const Eigen::MatrixXd &m,int *x, int *y);

};
#endif
