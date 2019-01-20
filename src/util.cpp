//Title : Fingerprints
//date : 07/01/2019
//Name : Li Jiajie - Ghimire Suraj - Bai Yuxin


#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <Eigen/Dense>
#include <fstream>

using namespace Eigen;
using namespace cv;
using namespace std;


double positive(double a) {
	if (a<0) return 0;
	else return a;
}

//calculate distance between point(x1,y1) and (x2,y2)
double distance(int x1, int y1, int x2, int y2){
	return sqrt((((x1-x2)*(x1-x2))+((y1-y2)*(y1-y2))));
}

//transport an image to a matrix
MatrixXd getMatrix(const Mat &img) {
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
Mat getImage(const MatrixXd &m) {
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
double getLongAxis(const Mat &image,int xc, int yc) {
	double a1 = 0;
	double a2 = 0;
	for(int i=0;i<xc;i++) {
		double d1 = distance(i,yc,xc,yc);
		if((d1>a1) & (image.at<uchar>(i,yc,0)==0)) {
			a1 = d1;
		}
	}
	for(int i=image.rows-1;i>xc;i--) {
		double d2 = distance(i,yc,xc,yc);
		if((d2>a2) & (image.at<uchar>(i,yc,0)==0)) {
			a2 = d2;
		}
	}
	return (a1+a2);
}

//get minor axis of ellipse
double getShortAxis(const Mat &image,int xc, int yc) {
	double b = 0;
	for(int j=0;j<image.cols;j++) {
		double d = distance(xc,j,xc,yc);   // d as distance between center and circumfurence
		if((d>b) & (image.at<uchar>(xc,j,0)==0)) {
			b = d;
		}
	}
	return b;
}


MatrixXd bilinear_inter(const MatrixXd &m) {
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


double fx(int i,int j,MatrixXd &m) {
	return (m(i+1,j)-m(i-1,j))/2;
}

double fy(int i,int j,MatrixXd &m) {
	return (m(i,j+1)-m(i,j-1))/2;
}

double fxy(int i,int j,MatrixXd &m) {
	double a = (m(i+1,j+1)-m(i-1,j+1))/2;
	double b = (m(i+1,j-1)-m(i-1,j-1))/2;
	return (a-b)/2;
}
/*
void bicubic_inter(MatrixXd &m) {
	Matrix4d A;
	Matrix4d C;
	A << 1,0,0,0,
	     0,0,1,0,
			 -3,3,-2,-1,
			 2,-2,1,1;
	C << 1,0,-3,2,
			 0,0,3,-2,
			 0,1,-2,1,
			 0,0,-1,1;
	for(int i=2;i<m.rows()-2;i++) {
		for(int j=2;j<m.cols()-2;j++) {
			if (m(i,j)==1) {
				Matrix4d B;
				B<<m(i-1,j-1),m(i-1,j+1),fy(i-1,j-1,m),fy(i-1,j+1,m),
				   m(i+1,j-1),m(i+1,j+1),fy(i+1,j-1,m),fy(i+1,j+1,m),
					 fx(i-1,j-1,m),fx(i-1,j+1,m),fxy(i-1,j-1,m),fxy(i-1,j+1,m),
					 fx(i+1,j-1,m),fx(i+1,j+1,m),fxy(i+1,j-1,m),fxy(i+1,j+1,m);
				if(i==2&j==2) cout<<"B="<<B<<endl;
				Matrix4d coeff = A*B*C;

				Vector4d x(1,i,i*i,i*i*i);
				Vector4d y(1,j,j*j,j*j*j);
				double val = x.transpose()*coeff*y;
				if(val>1) {
					val = 1;
				}
				m(i,j) = val;
			}
		}
	}
}
*/
double power(double a, double b) {
	if(b>=0) {
		return pow(a,b);
	}
	else {
		return 0;
	}
}

//interpolation
void bicubic_inter(MatrixXd &m) {
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
	/*
	if(val>1) {
		val = 1;
	}
	if(val<0) val=0;
	m(i,j) = val;
	*/
}

//rotation 10th January
MatrixXd rotation(const MatrixXd &m, double angle,int xc,int yc){
	angle = M_PI*angle/180;
	MatrixXd rm(m.rows(),m.cols());
	for(int i=0;i<rm.rows();i++){
		for(int j=0;j<rm.cols();j++){
			rm(i,j)=-1;
		}
	}
	for(int i=0;i<rm.rows();i++){
		for(int j=0;j<rm.cols();j++){
				int ip = (int)(cos(angle)*(i-xc) - sin(angle)*(j-yc)+xc);
				int jp = (int)(sin(angle)*(i-xc) +cos(angle)*(j-yc)+yc);
				if((ip>=0) & (ip<rm.rows()) & (jp>=0) & (jp<rm.cols())) {
					rm(ip,jp) = m(i,j);
				}
		}
	}
  //return bilinear_inter(rm);
	return rm;
}

MatrixXd translation(const MatrixXd &m, int a,int b) {
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

MatrixXd euclidien(const MatrixXd &m, double angle,int a,int b,int xc,int yc) {
	angle = M_PI*angle/180;
	MatrixXd rm(m.rows(),m.cols());
	for(int i=0;i<rm.rows();i++){
		for(int j=0;j<rm.cols();j++){
			rm(i,j)=1;
		}
	}
	for(int i=0;i<rm.rows();i++){
		for(int j=0;j<rm.cols();j++){
				int ip = (int)(cos(angle)*(i-xc) - sin(angle)*(j-yc)+xc)+a;
				int jp = (int)(sin(angle)*(i-xc) +cos(angle)*(j-yc)+yc)+b;
				if((ip>=0) & (ip<rm.rows()) & (jp>=0) & (jp<rm.cols())) {
					rm(ip,jp) = m(i,j);
				}
		}
	}
	bilinear_inter(rm);
  return rm;
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

double findDegree(const MatrixXd &m1,double d,int xc, int yc) {
	VectorXd axis((int)(360/d));
	for(int i=0;i*d<360;i++) {
		MatrixXd m3 = rotation(m1,i*d,xc,yc);
		axis(i) = getLongAxis(getImage(m3),xc,yc);
	}
	//cout<<"degree="<<diff<<endl;
	MatrixXf::Index max;
  axis.maxCoeff(&max);
	return (max*d-180);
}

//find position to translation
void findPosition(const MatrixXd &m1,const MatrixXd &m2,int *a,int *b) {
	MatrixXd pos(m1.rows(),m1.cols());
	for(int i=0;i<m1.rows();i++) {
		for(int j=0;j<m1.cols();j++) {
			pos(i,j) = 1000;
		}
	}
	for(int i=0;i<m1.rows();i=i+3) {
		for(int j=0;j<m1.cols();j=j+3) {
			MatrixXd m3 = translation(m1,i,j);
			pos(i,j) = (m3-m2).norm();
			//cout<<"norm = "<<pos(i,j)<<endl;
		}
	}
	//cout<<"degree="<<diff<<endl;
	MatrixXf::Index minRow, minCol;
  float min = pos.minCoeff(&minRow, &minCol);
	*a = minRow;
	*b = minCol;
	cout<<"a = "<<*a<<"b = "<<*b<<endl;
	cout<<"min = "<<min<<endl;
}

void getInfo(const MatrixXd &m,double *axisL,double *axisS,int *xc,int *yc) {
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
	cout<<"rows = "<<m.rows()<<endl;
	cout<<"cols = "<<m.cols()<<endl;
	cout<<"a1 = "<<a1<<endl;
	cout<<"a2 = "<<a2<<endl;
	cout<<"b1 = "<<b1<<endl;
	cout<<"b2 = "<<b2<<endl;
	cout<<"xc = "<<*xc<<endl;
	cout<<"yc = "<<*yc<<endl;
}

void range(Mat img) {
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

Mat test1(Mat &image) {
	Mat img = image;
	for(int i=0;i<img.rows/2;i++){
		for(int j=0;j<img.cols/2;j++){
				img.at<uchar>(i,j,0)=255;
				img.at<uchar>(img.rows-i,img.cols-j,0)=0;
		}
	}
	return img;
}

//stater 1
//symetrie
Mat sym_x(Mat &image) {
	Mat img = image.clone();
	for(int i=0;i<img.rows;i++){
		for(int j=0;j<img.cols;j++){
			img.at<uchar>(img.rows-i,j,0)=image.at<uchar>(i,j,0);
		}
	}
	return img;
}
Mat sym_y(Mat &image) {
	Mat img = image.clone();
	for(int i=0;i<img.rows;i++){
		for(int j=0;j<img.cols;j++){
			img.at<uchar>(i,j,0)=image.at<uchar>(i,img.cols-j,0);
		}
	}
	return img;
}
Mat sym_x_diag(Mat &image) {
	Mat img = image.clone();
	return img.t();
}
Mat sym_y_diag(Mat &image) {
	Mat img = image.clone();
	for(int i=0;i<img.rows;i++){
		for(int j=0;j<img.cols;j++){
			img.at<uchar>(i,j,0)=image.at<uchar>(img.rows-i,img.cols-j,0);
		}
	}
	img = img.t();
	return img;
}


double function_c1 (double r,double c){
	return exp(c*r);
}

double function_c2 (double r,double c,double t){
	if (r<t) {
		return exp(c*r);
	}
	else return 255;
}

double function_c3 (double r){
	return exp(-r*r*r);
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

Mat decrease(Mat &image, int xc, int yc,double *c,double a,double b,int n){
	Mat img = image.clone();
	for(int i=0;i<img.rows;i++){
		for(int j=0;j<img.cols;j++){
			int section = getSection(i,j,n,xc,yc);
			//cout<<"section = "<<section<<endl;
			if((section<0) | (section>=n) ){
				cout<<"section = "<<section<<endl;
			}

			double d = distance(xc,yc,i,j);
			/*
			if(d>t[section]) {
				//cout<<"d = "<<d<<endl;
				//cout<<"t = "<<t[section]<<endl;
				img.at<uchar>(i,j,0) = 255;
			}
			*/
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

double calcul_t(const Mat &img1, const Mat &img2,int xc,int yc,double degre1, double degre2){

	degre1 = 2*degre1*M_PI/360;
	degre2 = 2*degre2*M_PI/360;
	int n = 0;
	double s = 0;
//	double * array_c = new double[n];
	for(int i=0;i<img1.rows;i++){
		for(int j=0;j<img1.cols;j++){
			double x = i - xc;
			double y = j - yc;
			double r = sqrt((x*x+y*y));
			double theta = atan2(y,x);
			if((theta>=degre1)&(theta<=degre2)){
				double i2 = (int)img2.at<uchar>(i,j,0);
				double i1 = (int)img1.at<uchar>(i,j,0);
				if( (distance(xc,yc,i,j)!=0) & (i2==255) & (i1!=255)){
					double t = distance(xc,yc,i,j);
					s = s+t;
					n++;
				}
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

double *getTlist(const Mat &img1, const Mat &img2,int xc,int yc,int num) {
	double *t = new double[num];
	for(int i=0;i<num;i++) {
		t[i] = calcul_t(img1, img2,xc,yc,i*360/num-180, (i+1)*360/num-180);
		//cout<<"t = "<<t[i]<<endl;
	}
	return t;
}



int main( int argc, char** argv )
{
  Mat image = imread("clean_finger.png",CV_LOAD_IMAGE_GRAYSCALE);


	//Mat image2 = imread("weak_finger.png",CV_LOAD_IMAGE_GRAYSCALE);
	Mat image3 = imread("warp1_finger.png",CV_LOAD_IMAGE_GRAYSCALE);
	//int xc = (int)(image.rows/2);
	//int yc = (int)(image.cols/2);
	//double c = calcul_c(image, image2,xc,yc,2*360/8, (2+1)*360/8);
	/*
	double *c = getClist(image, image2,xc,yc,100);
	double a = getLongAxis(image2,xc,yc);
	double b = getShortAxis(image2,xc,yc);*/
	/*
	cout<<"a = "<<a<<endl;
	cout<<"b = "<<b<<endl;
	cout<<"rows = "<<image.rows<<endl;
	cout<<"cols = "<<image.cols<<endl;
	*/
	//cout<<"t = "<<c<<endl;
  //Mat im3 = decrease(image, xc,yc,c,a,b,100);
  //cout<<im2<<endl;
	MatrixXd m1 = getMatrix(image);
	MatrixXd m2 = getMatrix(image3);

	//sym_x

	double *a1 = new double(0);
	double *b1 = new double(0);
 	int *xc1 = new int(0);
	int *yc1 = new int(0);
	//cout<<"degree = "<<degree<<endl;
	getInfo(m2,a1,b1,xc1,yc1);

	//double degree = findDegree(m2,1,*xc1,*yc1);
	MatrixXd m4 = rotation(m1,45,*xc1,*yc1);
	//double *a2 = new double(0);
	//double *b2 = new double(0);
	//int *xc2 = new int(0);
	//int *yc2 = new int(0);
	//getInfo(m4,a2,b2,xc2,yc2);
	//MatrixXd m5 = translation(rotation(m1, -degree,*xc1,*yc1),*xc1-*xc2,*yc1-*yc2);
	//MatrixXd m5 = rotation(m1, -degree,*xc1,*yc1);
	//MatrixXd m5 = euclidien(m1, -degree,*xc2-*xc1,*yc2-*yc1,*xc1,*yc1);
	/*
	double *c = getClist(image, image2,*xc1,*yc1,100);
	Mat im3 = decrease(image, *xc1,*yc1,c,*a1,*b1,100);
	int *x = new int(0);
	int *y = new int(0);
	*/
	//findPosition(m1,m2,x,y);
	//MatrixXd m5 = translation(m4,*x,*y);
	Mat im3 = getImage(m4);

  namedWindow( "Display Image", WINDOW_AUTOSIZE );
  imshow( "Display Image", im3);
  imwrite("./warp1.png",im3);
  waitKey(0);
  return 0;
}
