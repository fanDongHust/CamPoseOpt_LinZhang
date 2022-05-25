/*
The Ground Model example
*/

#include <iostream>
#include <cmath>
#include <string>
#include <vector>
#include <boost/format.hpp>
#include <typeinfo>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "sophus/se3.h"
#include "sophus/so3.h"
#include <g2o/core/base_binary_edge.h>
// #include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include "initKDT.h"
#include "surroundView.h"

using namespace std; 
// using namespace cv;

int main(int argc, char** argv)
{
	cout<<"------------------Initialize camera pose----------------"<<endl;
	Sophus::SE3 T_FG;
	Sophus::SE3 T_LG;
	Sophus::SE3 T_BG;
	Sophus::SE3 T_RG;
	initializePose(T_FG, T_LG, T_BG, T_RG);
	cout<<T_FG.matrix()<<endl;

	cout<<"---------------Initialize K-------------------------------"<<endl;
	Eigen::Matrix3d K_F;
	Eigen::Matrix3d K_L;
	Eigen::Matrix3d K_B;
	Eigen::Matrix3d K_R;
	initializeK(K_F, K_L, K_B, K_R);
	cout<<K_F<<endl;
	
	cout<<"--------------------Initialize D--------------------------"<<endl;
	Eigen::Vector4d D_F;
	Eigen::Vector4d D_L;
	Eigen::Vector4d D_B; 
	Eigen::Vector4d D_R;
	initializeD(D_F, D_L, D_B, D_R);
	cout<<D_F<<endl;
	
	cout<<"--------------------Load images--------------------------"<<endl;
	int img_index = 3322;
	boost::format img_path_template("./test_cases/ALL/%06d ");
	cout<<"Reading "<<(img_path_template%img_index).str()<<"F.jpg"<<endl;
	cv::Mat img_F = cv::imread((img_path_template%img_index).str()+ "F.jpg");
	cout<<"Reading "<<(img_path_template%img_index).str()<<"L.jpg"<<endl;
	cv::Mat img_L = cv::imread((img_path_template%img_index).str()+ "L.jpg");
	cout<<"Reading "<<(img_path_template%img_index).str()<<"B.jpg"<<endl;
	cv::Mat img_B = cv::imread((img_path_template%img_index).str()+ "B.jpg");
	cout<<"Reading "<<(img_path_template%img_index).str()<<"R.jpg"<<endl;
	cv::Mat img_R = cv::imread((img_path_template%img_index).str()+ "R.jpg");
	cout<<img_F.size<<endl;

	cout<<"--------------------Init K_G--------------------------"<<endl;
	int rows = 1000;
	int cols = 1000;
	double dX = 0.1;
	double dY = 0.1;
	double fx =  1/dX;
	double fy = -1/dY;
	cv::Mat K_G = cv::Mat::zeros(3,3,CV_64FC1);
	K_G.at<double>(0,0) = fx;
	K_G.at<double>(1,1) = fy;
	K_G.at<double>(0,2) = cols/2;
	K_G.at<double>(1,2) = rows/2;
	K_G.at<double>(2,2) =   1.0;
	cout<<K_G<<endl;
	
	cout<<"--------------------Add noise to T matrix--------------------------"<<endl;
	Eigen::Matrix<double,6,1>  V6;
	V6<<0.01, -0.01, 0.01, -0.01, 0.01, -0.01;
	T_FG = Sophus::SE3::exp(T_FG.log()+V6);
	T_LG = Sophus::SE3::exp(T_LG.log()+V6);
	T_BG = Sophus::SE3::exp(T_BG.log()+V6);
	T_RG = Sophus::SE3::exp(T_RG.log()+V6);
	
	Sophus::SE3 T_GF = T_FG.inverse();
	Sophus::SE3 T_GL = T_LG.inverse();
	Sophus::SE3 T_GB = T_BG.inverse();
	Sophus::SE3 T_GR = T_RG.inverse();
	
	cout<<"--------------------Project images on the ground--------------------------"<<endl;
	cv::Mat img_GF = project_on_ground(img_F,T_FG,K_F,D_F,K_G,rows,cols);
	cv::Mat img_GL = project_on_ground(img_L,T_LG,K_L,D_L,K_G,rows,cols);
	cv::Mat img_GB = project_on_ground(img_B,T_BG,K_B,D_B,K_G,rows,cols);
	cv::Mat img_GR = project_on_ground(img_R,T_RG,K_R,D_R,K_G,rows,cols);
	
	cout<<"--------------------Stitch the surround view image--------------------------"<<endl;
	cv::Mat img_G = generate_surround_view(img_GF,img_GL,img_GB,img_GR,rows,cols);
	
	int overlap_x,overlap_y,overlap_w,overlap_h;
	overlap_x = 650;
	overlap_y = 0;
	overlap_w = 350;
	overlap_h = 350;
	cv::Rect rectFR(overlap_x,overlap_y,overlap_w,overlap_h);
	cv::Rect rectFL(overlap_x,overlap_y,overlap_w,overlap_h);
	
	// update T_FG
	Eigen::Matrix<double,6,1> rhophi_F,rhophi_R;
	double lr = 1e-8;
	int iter_times = 30;
	
	for(int i=0;i<iter_times;i++)
	{
		cout<<"iter "<<i<<": "<<endl;
		
		cv::namedWindow("img_GF",0);
		cv::imshow("img_GF",img_GF(rectFR));
		cv::namedWindow("img_GL",0);
		cv::imshow("img_GL",img_GR(rectFR));
		
// 		adjust_pose(img_GF,img_GL,rhophi_F,rhophi_L,K_G,
// 				overlap_x, overlap_y, overlap_w,overlap_h,lr);
		adjust_pose(img_GF,img_GR,rhophi_F,rhophi_R,K_G,
				overlap_x, overlap_y, overlap_w,overlap_h,lr);
		cout<<"rhophi_F: "<<endl<<rhophi_F<<endl;
		cout<<"rhophi_R: "<<endl<<rhophi_R<<endl;
		
// 		T_GF = Sophus::SE3::exp(T_GF.log() - rhophi_F);
		T_GF = Sophus::SE3::exp(rhophi_F).inverse()*T_GF;
		T_FG = T_GF.inverse();
		
		
// 		T_GR = Sophus::SE3::exp(T_GR.log()-rhophi_R);
// 		T_RG = T_GR.inverse();
		
		cout<<T_FG<<endl;
		cout<<T_RG<<endl;
		
		img_GF = project_on_ground(img_F,T_FG,K_F,D_F,K_G,rows,cols);
		img_GR = project_on_ground(img_R,T_RG,K_R,D_R,K_G,rows,cols);
		
		cv::waitKey(0);
	}
	
	T_GF = Sophus::SE3::exp(T_GF.log() - rhophi_F);
// 	T_GF = Sophus::SE3::exp(rhophi_F).inverse()*T_GF;
	T_FG = T_GF.inverse();
	
	// project img_GF with new T_FG
	img_GF = project_on_ground(img_F,T_FG,K_F,D_F,K_G,rows,cols);
	
	// Show new diff between img_GF and img_GL
	cv::Mat img_GF_gray,img_GL_gray;
	cv::cvtColor(img_GF(rectFL),img_GF_gray,cv::COLOR_BGR2GRAY);
	cv::medianBlur(img_GF_gray,img_GF_gray,5);
	img_GF_gray.convertTo(img_GF_gray, CV_64FC1);
	cv::cvtColor(img_GL(rectFL),img_GL_gray,cv::COLOR_BGR2GRAY);
	cv::medianBlur(img_GL_gray,img_GL_gray,5);
	img_GL_gray.convertTo(img_GL_gray, CV_64FC1);
	
	cv::Mat diff_FL,diff_FL_norm,diff_FL_color;
	cv::subtract(img_GF_gray,img_GL_gray,diff_FL,cv::noArray(),CV_64FC1);
	double coef = cv::mean(img_GF_gray).val[0]/cv::mean(img_GL_gray).val[0] ;
	cv::subtract(img_GF_gray,coef*img_GL_gray,diff_FL,cv::noArray(),CV_64FC1);
	cv::normalize(diff_FL,diff_FL_norm,0,255,cv::NORM_MINMAX,CV_8U);
	cv::applyColorMap(diff_FL_norm,diff_FL_color, cv::COLORMAP_JET);
	cv::namedWindow("diff_FL2",0);
	cv::imshow("diff_FL2",diff_FL_color);
	
	img_GF_gray.convertTo(img_GF_gray, CV_8U);
	cv::namedWindow("img_GF2",0);
	cv::imshow("img_GF2",img_GF_gray);
	
// 	cout<<multi_GF_x<<endl;
// 	cout<<multi_GF_x_norm<<endl;
	cv::waitKey(0);
	cv::destroyAllWindows();
	
	
// 	cv::Mat media_GF;
// // 	media_GF = img_GF_gray;
// 	cv::medianBlur(img_GF_gray,media_GF,3);
// 	cv::namedWindow("image0",CV_GUI_EXPANDED|CV_WINDOW_NORMAL);
// 	cv::imshow("image0",media_GF);
// 	
// 	cv::Mat grad_GF_x,grad_GF_x_norm,grad_GF_x_color;
// // 	cv::Sobel(media_GF, grad_GF_x, CV_64FC1, 1, 0, 7, 1, 0, cv::BORDER_DEFAULT);
// 	cv::Scharr(media_GF, grad_GF_x, CV_64FC1, 1, 0, 1, 0, cv::BORDER_DEFAULT);
// 	cv::normalize(grad_GF_x,grad_GF_x_norm,0,255,cv::NORM_MINMAX,CV_8U);
// 	cv::applyColorMap(grad_GF_x_norm,grad_GF_x_color, cv::COLORMAP_JET);
// 	
// 	cv::Mat grad_GF_y,grad_GF_y_norm,grad_GF_y_color;
// // 	cv::Sobel(media_GF, grad_GF_y, CV_64FC1, 0, 1, 7, 1, 0, cv::BORDER_DEFAULT);
// 	cv::Scharr(media_GF, grad_GF_y, CV_64FC1, 0, 1, 1, 0, cv::BORDER_DEFAULT);
// 	cv::normalize(grad_GF_y,grad_GF_y_norm,0,255,cv::NORM_MINMAX,CV_8U);
// 	cv::applyColorMap(grad_GF_y_norm,grad_GF_y_color, cv::COLORMAP_JET);
// 	
// 	cv::Mat grad_GF_xy,grad_GF_xy_norm,grad_GF_xy_color;
// 	cv::Mat pow_GF_x,pow_GF_y;
// 	cv::pow(grad_GF_x,2,pow_GF_x);
// 	cv::pow(grad_GF_y,2,pow_GF_y);
// 	cv::sqrt(pow_GF_x+pow_GF_y,grad_GF_xy);
// // 	cv::addWeighted(grad_GF_x, 0.5, grad_GF_y, 0.5, 0, grad_GF_xy);
// 	cv::normalize(grad_GF_xy,grad_GF_xy_norm,0,255,cv::NORM_MINMAX,CV_8U);
// 	cv::applyColorMap(grad_GF_xy_norm,grad_GF_xy_color, cv::COLORMAP_JET);
// 	
// 	cv::namedWindow("image1",0);
// 	cv::imshow("image1",grad_GF_x_color);
// 	cv::namedWindow("image2",0);
// 	cv::imshow("image2",grad_GF_y_color);
// 	cv::namedWindow("image3",0);
// 	cv::imshow("image3",grad_GF_xy_color);
// 	
// // 	cout<<img_GF_gray<<endl;
// 	cv::waitKey(0);
// 	cv::destroyAllWindows();
	
// 	cout<<"test"<<endl;
// 	cv::namedWindow("image4",0);
// 	cv::imshow("image4",img_GF(rect));
// 	cv::namedWindow("image5",0);
// 	cv::imshow("image5",img_GL(rect));
// 	cv::namedWindow("image6",0);
// 	cv::Mat absdiff_FL;
// 	cv::absdiff(img_GF(rect),img_GL(rect),absdiff_FL);
// 	cv::imshow("image6",absdiff_FL);
// 	cv::waitKey(0);
// 	cv::destroyAllWindows();
	return 0;
}