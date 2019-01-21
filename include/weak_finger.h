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
	static double function_c1 (double r,double c);
	static int getSection(int i,int j,int n,int xc,int yc);
	static double calcul_c(const Mat &img1, const Mat &img2,int xc,int yc,double degre1, double degre2);
	static double *getClist(const Mat &img1, const Mat &img2,int xc,int yc,int num);

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
	Mat static weakFinger(Mat &image, int xc, int yc,double *c,double a,double b,int n);
};

#endif
