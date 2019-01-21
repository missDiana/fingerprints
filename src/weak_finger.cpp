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

double distance(int x1, int y1, int x2, int y2){
	return sqrt((((x1-x2)*(x1-x2))+((y1-y2)*(y1-y2))));
}

double function_c1 (double r,double c){
	return exp(c*r);
}

int getSection(int i,int j,int n,int xc,int yc) {

			double x = i - xc;
			double y = j - yc;
			double r = sqrt((x*x+y*y));
			double theta = atan2(y,x) + M_PI;
			int k = (int)(n*theta/(2*M_PI));
			if (k==n) k = n-1;
			return k;
	}

Mat weakFinger(Mat &image, int xc, int yc,double *c,double a,double b,int n){
	Mat img = image.clone();
	for(int i=0;i<img.rows;i++){
		for(int j=0;j<img.cols;j++){
			int section = getSection(i,j,n,xc,yc);
			//cout<<"section = "<<section<<endl;
			if((section<0) | (section>=n) ){
				cout<<"section = "<<section<<endl;
			}
			double d = distance(xc,yc,i,j);
			double x = i - xc;
			double y = j - yc;
			double r = sqrt((x*x+y*y));
			double theta = atan2(y,x);
			double ellipse = a*b/(sqrt(b*b*cos(theta)*cos(theta)+a*a*sin(theta)*sin(theta)));
			if(r>ellipse) {
				img.at<uchar>(i,j,0) = 255;
			}
			else {
				int val = (int)(function_c1(d,c[section]) * image.at<uchar>(i,j,0));
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

double calcul_c(const Mat &img1, const Mat &img2,int xc,int yc,double degre1, double degre2){

	degre1 = 2*degre1*M_PI/360;
	degre2 = 2*degre2*M_PI/360;
	int n = 0;
	double s = 0;
	for(int i=0;i<img1.rows;i++){
		for(int j=0;j<img1.cols;j++){
			double x = i - xc;
			double y = j - yc;
			double r = sqrt((x*x+y*y));
			double theta = atan2(y,x);
			if((theta>=degre1)&(theta<=degre2)){
				//cout<<"d1 = "<<degre1<<", d2 = "<<degre2<<", d = "<<theta<<endl;
				double i2 = (int)img2.at<uchar>(i,j,0);
				double i1 = (int)img1.at<uchar>(i,j,0);
				if( (distance(xc,yc,i,j)!=0) & (i2!=0) & (i1!=0)){
					//cout <<"bizhi = "<<i2/i1<<endl;
					//cout<<"juli = "<<distance(xc,yc,i,j)<<endl;
					double c = (log(i2/i1))/distance(xc,yc,i,j);
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


double *getClist(const Mat &img1, const Mat &img2,int xc,int yc,int num) {
	double *c = new double[num];
	for(int i=0;i<num;i++) {
		c[i] = calcul_c(img1, img2,xc,yc,i*360/num-180, (i+1)*360/num-180);
		//cout<<"c = "<<c[i]<<endl;
	}
	return c;
}
