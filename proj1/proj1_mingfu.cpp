/*
Author: Mingfu Li
CS 5390 Proj 1
*/
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

using namespace cv;
using namespace std;

int alpha;
int sigma1, sigma2;
int width1, width2;
Mat dst1,dst2,final;
vector<Mat> planes1,planes2;

void getGaussianLow(Mat& kernel, double sigma, int width) {
    double center = width/2;
    double sum = 0.0; 

    for (int x = 0; x < width; ++x) {
        for (int y = 0; y < width; ++y) {
            kernel.at<float>(x,y) = exp( -1 * (pow((x-center)/sigma, 2.0) + pow((y-center)/sigma,2.0))/ (2 * M_PI * sigma * sigma));
            sum += kernel.at<float>(x,y);
        }
    }

    // normalize
    for (int x = 0; x < width; ++x) {
        for (int y = 0; y < width; ++y) {
            kernel.at<float>(x,y)/= sum;    
        }
    }
}

// helper functions
void cvt2planes(Mat image, vector<Mat>& planes) {
    Mat temp;
    cvtColor(image, temp, COLOR_BGR2HSV);
    split( temp, planes);
    // namedWindow("V", WINDOW_AUTOSIZE); 
    // imshow("V", input_planes[2]);
}

void cvt2image(Mat& dst, Mat p1, Mat p2, Mat p3) {
    Mat temp;
    vector<Mat> channels = {p1, p2, p3};
    merge(channels, temp);
    cvtColor(temp, dst, COLOR_HSV2BGR);
}

static void on_lowpass(int, void* ) 
{
    Mat g_kernel = Mat::zeros(width1+3,width1+3, CV_32F);
    Mat new_plane;

    getGaussianLow(g_kernel,sigma1+1, width1+3);
    // cout << "LPF" << endl << " " << g_kernel << endl << endl;
    filter2D( planes1[2], new_plane, -1, g_kernel);

    cvt2image(dst1, planes1[0], planes1[1], new_plane);
    imshow("LowPass", dst1);
}

static void on_highpass(int, void* ) 
{
    Mat g_kernel = Mat::zeros(width2+3,width2+3, CV_32F);
    Mat new_plane;

    getGaussianLow(g_kernel,sigma2+1, width2+3);
    filter2D( planes2[2], new_plane, -1, g_kernel); 

    // deduct low freq component from original image
    // new_plane = planes2[2] - new_plane;
    addWeighted( planes2[2], 1, new_plane, -1, 0.0, new_plane);

    cvt2image(dst2, planes2[0], planes2[1], new_plane);
    imshow("HighPass", dst2);
}

static void on_trackbar( int, void* )
{	 
    float alpha_float = (float)alpha/10;
    // cout << alpha_float << endl;
    float beta = 1.0 - alpha_float;

    // Using binear interpolation to blend
    addWeighted( dst2, alpha_float, dst1, beta, 0.0, final);
    imshow( "Blended", final );
}

int main( int argc, const char** argv ){
    Mat m1 = imread("dog.jpg");
    Mat m2 = imread("cat.jpg");

    dst1 = m1;
    dst2 = m2;

    namedWindow("LowPass", WINDOW_AUTOSIZE); 
    namedWindow("HighPass", WINDOW_AUTOSIZE); 
    moveWindow("HighPass", m1.cols, 0);
    namedWindow("Blended", WINDOW_AUTOSIZE); 
    moveWindow("Blended", m1.cols+m2.cols, 0);
    
    imshow("LowPass",m1);
    imshow("HighPass",m2);
    imshow("Blended",m1);

    createTrackbar( "Kernel Size","LowPass", &width1, 6, on_lowpass); 
    createTrackbar( "Sigma","LowPass", &sigma1, 3, on_lowpass); 

    createTrackbar( "Kernel Size","HighPass", &width2, 12, on_highpass); 
    createTrackbar( "Sigma","HighPass", &sigma2, 3, on_highpass); 
    
    createTrackbar( "Alpha", "Blended", &alpha,10, on_trackbar);


    cvt2planes(m1, planes1);
    cvt2planes(m2, planes2);

    int k = waitKey(0);

    if (k == 27) {
        destroyAllWindows();
    } else if (k == int('s')) {
        imwrite("low-pass-image.png", dst1);
        imwrite("high-pass-image.png", dst2);
        imwrite("blended-image.png", final);

    }
    return 0;
}
