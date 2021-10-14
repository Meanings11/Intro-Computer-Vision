/**
 * Mingfu Li
 * Project 2
 * Reference: @brief Demo code for detecting corners using Harris-Stephens method @author OpenCV team
 */

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

/// Global variables
Mat src, src_gray;
int src_width;
int thresh = 200;
int max_thresh = 255;

const char* source_window = "Source image";
const char* corners_window = "Corners detected";

/// Function header
void cornerHarris_demo( int, void* );
bool nms(int,int,Mat);
void featureMatch(vector<Point>&, vector<Point>&, Mat&);
int computeSSD(Point, Point);
Mat get5x5Patch(Point);

/**
 * @function main
 */
int main( int argc, char** argv )
{
    /// Load source image and convert it to gray
    Mat cmp;
    // src = imread( "triangle1.jpg");
    // cmp = imread( "triangle2.jpg");
    // src = imread( "yosemite1.jpg");
    // cmp = imread( "yosemite2.jpg");
    src = imread( "glass1.jpg");
    cmp = imread( "glass2.jpg");
    // src = imread( "img1.png");
    // cmp = imread( "img2.png");
    // cmp = imread( "img4.png");

    if ( src.empty() || cmp.empty() )
    {
        cout << "Could not open or find the image!\n" << endl;
        return -1;
    }

    src_width = src.cols;
    hconcat(src,cmp,src);

    /// Create a window and a trackbar
    namedWindow( source_window );
    createTrackbar( "Threshold: ", source_window, &thresh, max_thresh, cornerHarris_demo );


    // Study corners
    imshow( source_window, src );
    cvtColor( src, src_gray, COLOR_BGR2GRAY );
    cornerHarris_demo( 0, 0 );

    waitKey();

    return 0;
}

/**
 * @function cornerHarris_demo
 * @brief Executes the corner detection and draw a circle around the possible corners
 */
void cornerHarris_demo( int, void* )
{
    /// Detector parameters
    int blockSize = 2;
    int apertureSize = 3;
    double k = 0.04;
    int fpcnt=0;

    /// FEATURE DETECTOR:  Detecting corners
    Mat dst = Mat::zeros( src.size(), CV_32FC1 );
    Mat dst_eig = Mat::zeros( src.size(), CV_32FC1 );
    Mat dst_eigvecs = Mat::zeros( src.size(), CV_32FC(6) );
    cornerHarris( src_gray, dst, blockSize, apertureSize, k );
    cornerMinEigenVal( src_gray, dst_eig, blockSize, apertureSize );
    cornerEigenValsAndVecs( src_gray, dst_eigvecs, blockSize, apertureSize );

    // imshow("cornerHarris src_gray", src_gray);

    /// Normalizing
    Mat dst_norm, dst_norm_scaled;
    Mat draw_src = src.clone();
    normalize( dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
    convertScaleAbs( dst_norm, dst_norm_scaled );

    // imshow("cornerHarris norm  ", dst_norm_scaled);
    // moveWindow("cornerHarris norm  ", 0, src.rows);

    vector<Point> src_corners;
    vector<Point> cmp_corners;

    cout << " =========> APPLY THRESHOLD TO CORNERNESS MAP c(H)   " << thresh << endl;
    /// Drawing a circle around corners
    for( int i = 0; i < dst_norm.rows ; i++ )
    {
        for( int j = 0; j < dst_norm.cols; j++ )
        {
            if( (int) dst_norm.at<float>(i,j) > thresh )    // apply thresholding by user
            {
                if (nms(j,i,dst_norm)) {    // Non-maximal Suppression
                    circle( draw_src, Point(j,i), 5,  Scalar(0), 2, 8, 0 );
                    cout << "FEATURE POINT :" << ++fpcnt << endl;
                    cout << j << ", " << i << endl;
                    // cout << "c(H) " << dst.at<float>(i,j) << endl;
                    // cout << "Normalized c(H) " << dst_norm.at<float>(i,j) << endl;
                    // cout << "MinEigenValues " << dst_eig.at<float>(i,j) << endl;
                    // cout << "EigenVectors " << dst_eigvecs.at<Vec6f>(i,j) << endl;
                    cout << endl;

                    //Seperate features
                    if (j < src_width){
                        src_corners.push_back(Point(j,i));
                    } else {
                        cmp_corners.push_back(Point(j,i));
                    }
                }
            }
        }
    }
    
    cout << "Number of feature point on the left: " << src_corners.size() << endl;
    cout << "Number of feature point on the right: " << cmp_corners.size() << endl;
    cout << endl;

    //Apply Feature Matching
    featureMatch(src_corners, cmp_corners, draw_src);
    imshow( "Feature Matched", draw_src);
    moveWindow( "Feature Matched",0,src.rows);
}

//simple Non-maximal Suppression for 7x7 neighbour
bool nms(int x, int y, Mat img) {
    Mat patch = Mat(7,7, img.type());

    // Padding: ignore corner near boundary
    if (x - 3 < 0 || y -3 < 0 || x + 4 > img.cols || y+4 > img.rows) {
        return false;
    }

    Mat temp = img(Rect(x-3,y-3,7,7));
    temp.copyTo(patch);

    float curr = img.at<float>(y,x);

     for( int i = 0; i < patch.rows ; i++ )
    {
        for( int j = 0; j < patch.cols; j++ )
        {
            if (curr < patch.at<float>(i,j)) {
                return false;
            }
        }
    }
    return true;
    // cout << patch << endl;
}

void featureMatch(vector<Point>& src_corners, vector<Point>& cmp_corners, Mat& output) {
    // 2D array for storing all SSD score 
    vector<vector<int>> SSD;

    for( size_t i = 0; i < cmp_corners.size(); i++ )
    { 
        vector<int> sums;
        for( size_t j = 0; j < src_corners.size(); j++ )
        {
            int s = computeSSD(src_corners[j],cmp_corners[i]);
            sums.push_back(s);

            // threshold points
            if ( s < 10000) {
                line(output,src_corners[j],cmp_corners[i],Scalar(0,0,255),2);
                cout << src_corners[j] << " - " << cmp_corners[i] << " = "<< s << endl;
            }
        }
        SSD.push_back(sums);
    }
    //    cout << SSD.size() << endl;
}

int computeSSD(Point p1, Point p2) {
    Mat e1 = get5x5Patch(p1);
    Mat e2 = get5x5Patch(p2);

    int sum = 0, diff = 0;
    for (int i = 0; i < e1.cols; i ++) {
        for(int j = 0; j < e1.rows; j++) {
            diff =  (int)e1.at<uchar>(j,i) - (int)e2.at<uchar>(j,i);
            sum += pow(diff,2);
        }
    }
    return sum;
}

// get 5x5 feature descriptor
Mat get5x5Patch(Point p) {
    Mat patch = Mat(5,5, src_gray.type());
    Mat temp = src_gray(Rect(p.x-2,p.y-2,5,5));
    temp.copyTo(patch);
    return patch;
}