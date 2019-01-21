//Title : Fingerprints
//date : 07/01/2019
//Name : Li Jiajie - Ghimire Suraj - Bai Yuxin

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



//calculate distance between point(x1,y1) and (x2,y2)
double util::distance(int x1, int y1, int x2, int y2){
	return sqrt((((x1-x2)*(x1-x2))+((y1-y2)*(y1-y2))));
}

//transport an image to a matrix
MatrixXd util::getMatrix(const Mat &img) {
	MatrixXd m(img.rows,img.cols);
	for(int i=0;i<img.rows;i++){
		for(int j=0;j<img.cols;j++){
			m(i,j) = ((int)img.at<uchar>(i,j,0));
			m(i,j) = m(i,j)/255;
		}
	}
	return m;
}

//transport a matrix to an image
Mat util::getImage(const MatrixXd &m) {
	Mat img(m.rows(),m.cols(),CV_8UC1);
	for(int i=0;i<img.rows;i++){
		for(int j=0;j<img.cols;j++){
			int val = (int)(m(i,j)*255);
			img.at<uchar>(i,j,0) = (uchar)val;
		}
	}
	return img;
}

//main course 1 decrease an image
//get major axis of ellipse
double util::getMajorAxis(const MatrixXd &image) {
	int xc = image.rows()/2;
	int yc = image.cols()/2;
	double a1 = 0;
	double a2 = 0;
	for(int i=0;i<xc;i++) {
		double d1 = distance(i,yc,xc,yc);
		if((d1>a1) & (image(i,yc)==0)) {
			a1 = d1;
		}
	}
	for(int i=image.rows()-1;i>xc;i--) {
		double d2 = distance(i,yc,xc,yc);
		if((d2>a2) & (image(i,yc)==0)) {
			a2 = d2;
		}
	}
	return (a1+a2)/2;
}

//get minor axis of ellipse
double util::getMinorAxis(const MatrixXd &image) {
	int xc = image.rows()/2;
	int yc = image.cols()/2;
	double b1 = 0;
	double b2 = 0;
	for(int j=0;j<yc;j++) {
		double d1 = distance(xc,j,xc,yc);
		if((d1>b1) & (image(xc,j)==0)) {
			b1 = d1;
		}
	}
	for(int j=image.rows()-1;j>yc;j--) {
		double d2 = distance(xc,j,xc,yc);
		if((d2>b2) & (image(xc,j)==0)) {
			b2 = b2;
		}
	}
	return (b1+b2)/2;
}


MatrixXd util::bilinear_inter(const MatrixXd &m) {
	MatrixXd mp = m;
	for(int i=1;i<mp.rows()-1;i++){
		for(int j=1;j<mp.cols()-1;j++){
				if (mp(i,j)==-1) {
					Vector4d p;
					p << m(i-1,j-1), m(i-1,j+1), m(i+1,j-1), m(i+1,j+1);
					int n = 0;
					double s = 0;
					for (int k=0;k<4;k++) {
						if(p(k)>=0) {
							s = s + p(k);
							n++;
						}
					}
					if(n!=0) {
						mp(i,j) = s/n;
					}
				}
		}
	}
	for(int i=0;i<m.rows();i++){
		for(int j=0;j<m.cols();j++){
				if (mp(i,j)<0) {
					mp(i,j)=1;
				}
		}
	}
	return mp;
}


double util::fx(int i,int j,MatrixXd &m) {
	return (m(i+1,j)-m(i-1,j))/2;
}

double util::fy(int i,int j,MatrixXd &m) {
	return (m(i,j+1)-m(i,j-1))/2;
}

double util::fxy(int i,int j,MatrixXd &m) {
	double a = (m(i+1,j+1)-m(i-1,j+1))/2;
	double b = (m(i+1,j-1)-m(i-1,j-1))/2;
	return (a-b)/2;
}

double util::power(double a, double b) {
	if(b>=0) {
		return pow(a,b);
	}
	else {
		return 0;
	}
}

//interpolation
void util::bicubic_inter(MatrixXd &m) {
	MatrixXd A(16,16);
	VectorXd b(16);
	for(int i=2;i<m.rows()-2;i++) {
		for(int j=2;j<m.cols()-2;j++) {
			if (m(i,j)==1) {
				for (int x=0;x<4;x++) {
					for (int y=0;y<4;y++) {
						A(0,4*y+x) = power(i-1,x)*power(j-1,y);
						A(1,4*y+x) = power(i-1,x)*power(j+1,y);
						A(2,4*y+x) = power(i+1,x)*power(j-1,y);
						A(3,4*y+x) = power(i+1,x)*power(j+1,y);
						A(4,4*y+x) = x*power(i-1,x-1)*power(j-1,y);
						A(5,4*y+x) = x*power(i-1,x-1)*power(j+1,y);
						A(6,4*y+x) = x*power(i+1,x-1)*power(j-1,y);
						A(7,4*y+x) = x*power(i+1,x-1)*power(j+1,y);
						A(8,4*y+x) = y*power(i-1,x)*power(j-1,y-1);
						A(9,4*y+x) = y*power(i-1,x)*power(j+1,y-1);
						A(10,4*y+x) = y*power(i+1,x)*power(j-1,y-1);
						A(11,4*y+x) = y*power(i+1,x)*power(j+1,y-1);
						A(12,4*y+x) = x*y*power(i-1,x-1)*power(j-1,y-1);
						A(13,4*y+x) = x*y*power(i-1,x-1)*power(j+1,y-1);
						A(14,4*y+x) = x*y*power(i+1,x-1)*power(j-1,y-1);
						A(15,4*y+x) = x*y*power(i+1,x-1)*power(j+1,y-1);
					}
				}
				b(0) = m(i-1,j-1);
				b(1) = m(i-1,j+1);
				b(2) = m(i+1,j-1);
				b(3) = m(i+1,j+1);

				b(4) = fx(i-1,j-1,m);
				b(5) = fx(i-1,j+1,m);
				b(6) = fx(i+1,j-1,m);
				b(7) = fx(i+1,j+1,m);

				b(8) = fy(i-1,j-1,m);
				b(9) = fy(i-1,j+1,m);
				b(10) = fy(i+1,j-1,m);
				b(11) = fy(i+1,j+1,m);

				b(12) = fxy(i-1,j-1,m);
				b(13) = fxy(i-1,j+1,m);
				b(14) = fxy(i+1,j-1,m);
				b(15) = fxy(i+1,j+1,m);

				VectorXd X = A.colPivHouseholderQr().solve(b);
				double val;
				for (int x=0;x<4;x++) {
					for (int y=0;y<4;y++) {
						val = X(4*y+x)*pow(i,x)*pow(j,y);
					}
				}
			}
		}
	}
}



//Imaged tried of double size
/*MatrixXd rotation(const MatrixXd &m, double angle){
	angle = angle/180;
	MatrixXd rm(2*m.rows(),2*m.cols());
	for(int i=0;i<2*rm.rows();i++){
		for(int j=0;j<2*rm.cols();j++){
			rm(i,j)=1;
		}
	}
	for(int i=0;i<2*rm.rows();i++){
		for(int j=0;j<2*rm.cols();j++){
				int ip = (int)(cos(angle)*i - sin(angle)*j);
				int jp = (int)(sin(angle)*i +cos(angle)*j);
				if((ip>=0) & (ip<rm.rows()) & (jp>=0) & (jp<rm.cols())) {
					rm(ip,jp) = m(i,j);
				}
		}
	}
  return rm;
}*/

void util::getInfo(const MatrixXd &m,double *axisL,double *axisS,int *xc,int *yc) {
	int a1=2*m.rows();
	int a2=-1;
	int b1=2*m.cols();
	int b2=-1;
	for(int i=5;i<m.rows()-5;i++) {
		for(int j=5;j<m.cols()-5;j++) {
			if(m(i,j)==0) {
				if(i<a1) a1=i;
				if(i>a2) a2=i;
				if(j<b1) b1=j;
				if(j>b2) b2=j;
			}
		}
	}
	*axisL = (a2-a1)/2;
	*axisS = (b2-b1)/2;
	*xc = (int)((a1+a2)/2);
	*yc = (int)((b1+b2)/2);
}

void util::range(Mat img) {
	Scalar intensity = img.at<uchar>(0, 0);
	int min = (int)intensity.val[0];
	int max = (int)intensity.val[0];
	for(int i=0;i<img.rows;i++){
		for(int j=0;j<img.cols;j++){
			int val = (int)img.at<uchar>(i, j,0);
			//int val = (int)intensity.val[0];
			if(val>max) {
				max = val;
			}
			if(val<min) {
				min = val;
			}
		}
	}
	cout<<"Min = "<<min<<", Max = "<<max<<endl;
}

Mat util::changeBlock(Mat &image) {
	Mat img = image;
	for(int i=0;i<img.rows/2;i++){
		for(int j=0;j<img.cols/2;j++){
				img.at<uchar>(i,j,0)=255;
				img.at<uchar>(img.rows-i,img.cols-j,0)=0;
		}
	}
	return img;
}

Mat util::sym_x(Mat &image) {
	Mat img = image.clone();
	for(int i=0;i<img.rows;i++){
		for(int j=0;j<img.cols;j++){
			img.at<uchar>(img.rows-i,j,0)=image.at<uchar>(i,j,0);
		}
	}
	return img;
}
Mat util::sym_y(Mat &image) {
	Mat img = image.clone();
	for(int i=0;i<img.rows;i++){
		for(int j=0;j<img.cols;j++){
			img.at<uchar>(i,j,0)=image.at<uchar>(i,img.cols-j,0);
		}
	}
	return img;
}
Mat util::sym_x_diag(Mat &image) {
	Mat img = image.clone();
	return img.t();
}
Mat util::sym_y_diag(Mat &image) {
	Mat img = image.clone();
	for(int i=0;i<img.rows;i++){
		for(int j=0;j<img.cols;j++){
			img.at<uchar>(i,j,0)=image.at<uchar>(img.rows-i,img.cols-j,0);
		}
	}
	img = img.t();
	return img;
}
void util::getCenter(const MatrixXd &m,int *x, int *y) {
	double* a = new double(0);
	double* b = new double(0);
	util::getInfo(m,a,b,x,y);
}
