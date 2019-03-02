#include <util.h>
#include <warping.h>
#include <weak_finger.h>
#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <Eigen/Dense>
#include <fstream>
using namespace Eigen;
using namespace cv;
using namespace std;
//rotation 10th January
double warping::squeezFunction(int i,int j,int xc, int yc,double ratio,double k) {
	int x = i - xc;
	int y = j - yc;
	double r = sqrt((pow(x/ratio,2)+y*y));
	return exp(-pow(r,1)/k);
}
MatrixXd warping::rotation(const MatrixXd &m, double angle,int xc,int yc){
	angle = M_PI*angle/180;
	MatrixXd rm(m.rows(),m.cols());
	for(int i=0;i<rm.rows();i++){
		for(int j=0;j<rm.cols();j++){
			rm(i,j)=-1;
		}
	}
	for(int i=0;i<rm.rows();i++){
		for(int j=0;j<rm.cols();j++){
				int ip = round(cos(angle)*(i-xc) - sin(angle)*(j-yc)+xc);
				int jp = round(sin(angle)*(i-xc) +cos(angle)*(j-yc)+yc);
				if((ip>=0) & (ip<rm.rows()) & (jp>=0) & (jp<rm.cols())) {
					rm(ip,jp) = m(i,j);
				}
		}
	}
  	return util::bilinear_inter(m,rm,angle,xc,yc);
}

MatrixXd warping::translation(const MatrixXd &m, int a,int b) {
	MatrixXd rm(m.rows(),m.cols());
	for(int i=0;i<rm.rows();i++){
		for(int j=0;j<rm.cols();j++){
			rm(i,j)=1;
		}
	}
	for(int i=0;i<rm.rows();i++){
		for(int j=0;j<rm.cols();j++){
				int ip = i+a;
				int jp = j+b;
				if((ip>=0) & (ip<rm.rows()) & (jp>=0) & (jp<rm.cols())) {
					rm(ip,jp) = m(i,j);
				}
		}
	}
  return rm;
}

MatrixXd warping::euclidien(const MatrixXd &m, double angle,int a,int b,int xc,int yc) {
	angle = M_PI*angle/180;
	MatrixXd rm(m.rows(),m.cols());
	for(int i=0;i<rm.rows();i++){
		for(int j=0;j<rm.cols();j++){
			rm(i,j)=1;
		}
	}
	for(int i=0;i<rm.rows();i++){
		for(int j=0;j<rm.cols();j++){
				int ip = round(cos(angle)*(i-xc) - sin(angle)*(j-yc)+xc)+a;
				int jp = round(sin(angle)*(i-xc) +cos(angle)*(j-yc)+yc)+b;
				if((ip>=0) & (ip<rm.rows()) & (jp>=0) & (jp<rm.cols())) {
					rm(ip,jp) = m(i,j);
				}
		}
	}
	return util::bilinear_inter(m,rm,angle,xc,yc);
}

double warping::findDegree(const MatrixXd &m1,double d) {
	VectorXd axis(int(360/d));
	int xc = m1.rows()/2;
	int yc = m1.cols()/2;
	for(int i=0;i*d<360;i++) {
		MatrixXd m3 = rotation(m1,i*d,xc,yc);
		axis(i) = util::getMajorAxis(m3);
	}
	MatrixXf::Index max;
  	axis.maxCoeff(&max);
	return (max*d);
}

MatrixXd warping::squeezedRotation(const MatrixXd &m, double angle,int xc,int yc) {
	angle = M_PI*angle/180;
	MatrixXd rm(m.rows(),m.cols());
	double ratio = util::getMajorAxis(m)/util::getMinorAxis(m);
	cout<<"ratio = "<<ratio<<endl;
	double k = 60;
	for(int i=0;i<rm.rows();i++){
		for(int j=0;j<rm.cols();j++){
			rm(i,j)=-1;
		}
	}
	for(int i=0;i<rm.rows();i++){
		for(int j=0;j<rm.cols();j++){
			double theta = warping::squeezFunction(i,j,xc,yc,ratio,k)*angle;
			int ip = round(cos(theta)*(i-xc) - sin(theta)*(j-yc)+xc);
			int jp = round(sin(theta)*(i-xc) +cos(theta)*(j-yc)+yc);
			if((ip>=0) & (ip<rm.rows()) & (jp>=0) & (jp<rm.cols())) {
				rm(ip,jp) = m(i,j);
			}
		}
	}
  	return util::bilinear_inter_naive(rm);
}


Mat warping::warpingFinger(const Mat &img1, const Mat &img2) {
	MatrixXd m1 = util::getMatrix(img1);
	MatrixXd m2 = util::getMatrix(img2);
	int* x1 = new int(0);
	int* y1 = new int(0);
	int* x2 = new int(0);
	int* y2 = new int(0);
	util::getCenter(m1,x1,y1);
	cout<<"haha"<<endl;
	double d = findDegree(m2,1);
	util::getCenter(m2,x2,y2);
	MatrixXd m3 = translation(rotation(m1,d,*x1,*y1),*x1 - *x2,*y1 - *y2);
	return util::getImage(m3);
}
