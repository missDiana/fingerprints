#ifndef WEAK_FINGER_H
#define WEAK_FINGER_H
#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <Eigen/Dense>
#include <fstream>
using namespace Eigen;
using namespace cv;
using namespace std;
class weak_finger {
private:
	/**
	Weight function for the weak pressure approximation
	@param r The distance between (x,y) and the center
	@param c A parameter of the function
	*/
	static double function_c1 (double r,double k);

	/**
	Auxilary function to determine the direction that a pixel belongs to
	@param i Coordinate of the pixel
	@param j Coordinate of the pixel
	@param n Number of directions
	*/
	static int getSection(int i,int j,int n, const Mat &image);

	/**
	Auxilary function to determine the parameter of the weight
	function in a direction defined by the space between two lines in the polar system
	@param img1 The clean finger
	@param img2 The weak finger
	@param degree1 The coordinate theta in polar system of the first line
	@param degree2 The coordinate theta in polar system of the second line
	*/
	static double calcul_k(const Mat &img1, const Mat &img2,double degree1, double degree2);

	/**
	Auxilary function to determine the parameter of the weight
	function in different directions
	@param img1 The clean finger
	@param img2 The weak finger
	@param num The number of directions
	@TODO Try to avoid using complex type as return parameters. Pass it as an argument of your function with non const reference
	*/
	static double *getKlist(const Mat &img1, const Mat &img2,int num);

	/**
	 * Calculate the distance between two points
	 * @param x1 coodinate x for the first point
	 * @param y1 coodinate y for the first point
	 * @param x2 coodinate x for the second point
	 * @param y2 coodinate y for the second point
	 * @return The distance between two points
	 */
	double static distance(int x1, int y1, int x2, int y2);

public:
	/**
	Perform an approximation of a weak finger
	@param img1 The clean finger
	@param img2 The weak finger
	@param n Number of directions of anisotropy
	*/
	Mat static weakFinger(const Mat &img1, const Mat &img2, int n);
};

#endif
