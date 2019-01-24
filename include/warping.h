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

	/**
	* @param d is the multiplication by which below loop changes.
	* @param xc is the x part of the center of the image
	* @param yc is the y part of the center of the image
	* @param angle is the angle to which image is to be rotated
	* @return the image matrix.
	*/

	static double findDegree(const MatrixXd &m1,double d);




public:

	static double squeezFunction(int i,int j,int xc, int yc,double ratio,double k);
	/**
	* @param xc is the x part of the center of the image
	* @param yc is the y part of the center of the image
	* @param angle is the angle to which image is to be rotated
	* @return matrix generated after rotation of given angle.
	*/
	static MatrixXd rotation(const MatrixXd &m, double angle,int xc,int yc);
	/**
	* @param a is the translation in x coodrinates.
	* @param b is the translation in y coodrinates.
	* @return matrix generated after translation.
	*/
	static MatrixXd translation(const MatrixXd &m, int a,int b);

	/**
	* @param a is the translation in x coodrinates.
	* @param b is the translation in y coodrinates.
	* @param xc is the x part of the center of the image
	* @param yc is the y part of the center of the image
	* @param angle is the angle to which image is to be rotated
	* @return matrix generated after eucledian tranformation
	*/
	static MatrixXd euclidien(const MatrixXd &m, double angle,int a,int b,int xc,int yc);
	static MatrixXd squeezedRotation(const MatrixXd &m, double angle,int xc,int yc);
	static Mat warpingFinger(const Mat &img1, const Mat &img2);

};
#endif
