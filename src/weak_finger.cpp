#include "util.h"
#include "warping.h"
#include "weak_finger.h"
#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <Eigen/Dense>
#include <fstream>
using namespace Eigen;
using namespace cv;
using namespace std;

double weak_finger::function_c1 (double r,double k){
	return exp(c*k);
}

int weak_finger::getSection(int i,int j,int n, Mat &image) {
		double* xc = new double(0);
		double* yc = new double(0);
		getCenter(getMatrix(image),xc,yc);
		double x = i - *xc;
		double y = j - *yc;
		double r = sqrt((x*x+y*y));
		double theta = atan2(y,x) + M_PI;
		int k = (int)(n*theta/(2*M_PI));
		if (k==n) k = n-1;
		return k;
	}

Mat weak_finger::weakFinger(const Mat &img1, const Mat &img1, int n){
	Mat img = img1.clone();
	double* xc = new double(0);
	double* yc = new double(0);
	getCenter(getMatrix(image),xc,yc);
	double *k = getKlist(img1,img2,n);
	for(int i=0;i<img.rows;i++){
		for(int j=0;j<img.cols;j++){
			int section = getSection(i,j,n,*xc,*yc);
			//cout<<"section = "<<section<<endl;
			if((section<0) | (section>=n) ){
				cout<<"section = "<<section<<endl;
			}
			double d = distance(*xc,*yc,i,j);
			double x = i - *xc;
			double y = j - *yc;
			double r = sqrt((x*x+y*y));
			double theta = atan2(y,x);
			double ellipse = a*b/(sqrt(b*b*cos(theta)*cos(theta)+a*a*sin(theta)*sin(theta)));
			if(r>ellipse) {
				img.at<uchar>(i,j,0) = 255;
			}
			else {
				int val = (int)(function_c1(d,k[section]) * img1.at<uchar>(i,j,0));
				if((val>255)) {
					img.at<uchar>(i,j,0) = 255;
				}
				else {
					img.at<uchar>(i,j,0) = val;
				}
			}
		}
	}
	return img;
}

double weak_finger::calcul_k(const Mat &img1, const Mat &img2,double degree1, double degree2){
	double* xc = new double(0);
	double* yc = new double(0);
	getCenter(getMatrix(image),xc,yc);
	degree1 = 2*degree1*M_PI/360;
	degree2 = 2*degree2*M_PI/360;
	int n = 0;
	double s = 0;
	for(int i=0;i<img1.rows;i++){
		for(int j=0;j<img1.cols;j++){
			double x = i - *xc;
			double y = j - *yc;
			double r = sqrt((x*x+y*y));
			double theta = atan2(y,x);
			if((theta>=degree1)&(theta<=degree2)){
				//cout<<"d1 = "<<degre1<<", d2 = "<<degre2<<", d = "<<theta<<endl;
				double i2 = (int)img2.at<uchar>(i,j,0);
				double i1 = (int)img1.at<uchar>(i,j,0);
				if( (distance(*xc,*yc,i,j)!=0) & (i2!=0) & (i1!=0)){
					//cout <<"bizhi = "<<i2/i1<<endl;
					//cout<<"juli = "<<distance(xc,yc,i,j)<<endl;
					double c = (log(i2/i1))/distance(*xc,*yc,i,j);
					s = s+c;
					n++;
				}
			}
			else {
				//cout<<"d1 = "<<degre1<<", d2 = "<<degre2<<", d = "<<theta<<endl;
			}

		}
	}
	s = s/n;
	return s;
}


double * weak_finger::getKlist(const Mat &img1, const Mat &img2,int num) {
	double* xc = new double(0);
	double* yc = new double(0);
	getCenter(getMatrix(image),xc,yc);
	double *k = new double[num];
	for(int i=0;i<num;i++) {
		k[i] = calcul_k(img1, img2,*xc,*yc,i*360/num-180, (i+1)*360/num-180);
		//cout<<"c = "<<c[i]<<endl;
	}
	return k;
}