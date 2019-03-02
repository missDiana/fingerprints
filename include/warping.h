#ifndef WARPING_H
#define WARPING_H
#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <Eigen/Dense>
#include <fstream>

class warping {
private:

	/**
	* @param d is the multiplication by which below loop changes.
	* @param xc is the x part of the center of the image
	* @param yc is the y part of the center of the image
	* @param angle is the angle to which image is to be rotated
	* @return the image matrix.
	*/

	static double findDegree(const Eigen:MatrixXd &m1,double d);




public:

	static double squeezFunction(int i,int j,int xc, int yc,double ratio,double k);
	/**
	* @param xc is the x part of the center of the image
	* @param yc is the y part of the center of the image
	* @param angle is the angle to which image is to be rotated
	* @return matrix generated after rotation of given angle.
	*/
	static Eigen:MatrixXd rotation(const Eigen:MatrixXd &m, double angle,int xc,int yc);
	/**
	* @param a is the translation in x coodrinates.
	* @param b is the translation in y coodrinates.
	* @return matrix generated after translation.
	*/
	static Eigen:MatrixXd translation(const Eigen:MatrixXd &m, int a,int b);

	/**
	* @param a is the translation in x coodrinates.
	* @param b is the translation in y coodrinates.
	* @param xc is the x part of the center of the image
	* @param yc is the y part of the center of the image
	* @param angle is the angle to which image is to be rotated
	* @return matrix generated after eucledian tranformation
	*/
	static Eigen:MatrixXd euclidien(const Eigen:MatrixXd &m, double angle,int a,int b,int xc,int yc);
	static Eigen:MatrixXd squeezedRotation(const Eigen:MatrixXd &m, double angle,int xc,int yc);
	static cv::Mat warpingFinger(const cv::Mat &img1, const cv::Mat &img2);

};
#endif
