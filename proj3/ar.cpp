/*
Project 3 Augmented Reality
Mingfu Li
https://www.learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/
*/

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include "opencv2/core.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <iostream>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

// choose this boolean value to see image test results
const bool imageMode = false;
int img_object_col;


void drawCorners(Mat& dst, vector<Point2f> corners, Scalar color) {
    line(dst, corners[0], corners[1], color, 2);
    line(dst, corners[1], corners[2], color, 2);
    line(dst, corners[2], corners[3], color, 2);
    line(dst, corners[3], corners[0], color, 2);
}

void removeBlackTrim(Mat& src) {
    Mat gray, mask;
    cvtColor(src, gray, COLOR_BGR2GRAY);
    threshold( gray, mask, 1, 255, THRESH_BINARY);

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours( mask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE );
    vector<Rect> boundRect( contours.size() );

    // for (size_t i = 0; i < contours.size(); i++) {
         boundRect[0] = boundingRect( contours[0] );
    // }

    cout << boundRect[0].tl() << " " << boundRect[0].br() << endl;
}

int main( int argc, char* argv[] )
{
    Mat img_object = imread("cv_cover.jpg",0);
    Mat img_scene =  imread("cv_desk.png", 0);
    Mat img_scene_colored = imread("cv_desk.png");
    // Mat img_object = imread("box.png",0);
    // Mat img_scene =  imread("box_in_scene.png", 0);
    // Mat img_scene_colored = imread("box_in_scene.png");
    Mat img_clip = imread("hp_cover.jpg");
    VideoCapture cap("ar_source.mov"); 

    // Check if camera opened successfully
    if(!cap.isOpened()){
        cout << "Error opening video stream or file" << endl;
        return -1;
    }

   if ( img_object.empty() || img_scene.empty() || img_clip.empty() )
    {
        cout << "Error opening images!\n" << endl;
        return -1;
    }
 
    //Feature detection, description: SURF detector
    int minHessian = 400;
    Ptr<SURF> detector = SURF::create( minHessian );
    vector<KeyPoint> keypoints_object, keypoints_scene;
    Mat descriptors_object, descriptors_scene;
    detector->detectAndCompute( img_object, noArray(), keypoints_object, descriptors_object );
    detector->detectAndCompute( img_scene, noArray(), keypoints_scene, descriptors_scene );

    //Feature matching: FLANN matcher
    vector< vector<DMatch> > pre_matches;
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    matcher->knnMatch( descriptors_object, descriptors_scene, pre_matches, 2 );

    vector<DMatch> matches;
     for (size_t i = 0; i < pre_matches.size(); i++)
    {
            matches.push_back(pre_matches[i][0]);
    }

    //Find Homography
    vector<Point2f> obj;
    vector<Point2f> scene;
    for( size_t i = 0; i < matches.size(); i++ )
    {
        obj.push_back( keypoints_object[ matches[i].queryIdx ].pt );
        scene.push_back( keypoints_scene[ matches[i].trainIdx ].pt );
    }
    Mat H = findHomography( obj, scene, RANSAC );

    // Find object corners in scene
    vector<Point2f> obj_corners(4), scene_corners(4);
    obj_corners[0] = Point2f(0, 0);
    obj_corners[1] = Point2f( (float)img_object.cols, 0 );
    obj_corners[2] = Point2f( (float)img_object.cols, (float)img_object.rows );
    obj_corners[3] = Point2f( 0, (float)img_object.rows );
    perspectiveTransform( obj_corners, scene_corners, H);

    // Draw corners 
    Mat img_matches = img_scene.clone();
    cvtColor(img_matches, img_matches, cv::COLOR_GRAY2RGB);
    img_object_col = img_object.cols;

    drawCorners(img_matches, scene_corners, Scalar(0, 255, 0));
    for(int i = 0; i < scene_corners.size(); i++) {
        circle( img_matches, scene_corners[i], 5,  Scalar(255,0,0), 2, 8, 0 );
    }

    //Display matching
    imshow("Object", img_object);
    imshow("Object detection and Matching", img_matches );
    moveWindow("Object detection and Matching", img_object.cols, 0);

    // Image or video clipping
    Mat img_output = img_scene_colored;
    if (imageMode) {

        // Project the object
        Mat img_warped;
        resize(img_clip, img_clip, Size(img_object.cols, img_object.rows));
        warpPerspective(img_clip, img_warped, H, Size(img_output.cols, img_output.rows)); 

        //Apply mask in image insertion
        Mat mask;
        cvtColor(img_warped, mask, cv::COLOR_RGB2GRAY);
        img_warped.copyTo(img_output, mask);

        drawCorners(img_output, scene_corners, Scalar(0, 255, 0));

        // imshow("New Object", img_clip);
        imshow("New Object Projection", img_output );
        moveWindow("New Object Projection", img_object_col+img_matches.cols, 0);

    } else {
        // video clip
        while(1){

            Mat frame;
            Mat frame_warped;
            cap >> frame;
        
            // If the frame is empty, break immediately 
            if (frame.empty())
            break;

            //remove black trims of video frame
            Rect ROI(0, 50, 640, 260);
            Mat cropped_frame = frame(ROI);

            resize(cropped_frame, cropped_frame, Size(img_object.cols, img_object.rows), 0, 0, INTER_CUBIC);
            warpPerspective(cropped_frame, frame_warped, H, Size(img_output.cols, img_output.rows)); 
            // imshow("Frame", frame_warped );

            //Apply mask in frame insertion
            Mat mask;
            cvtColor(frame_warped, mask, cv::COLOR_RGB2GRAY);
            frame_warped.copyTo(img_output, mask);

            // drawCorners(img_output, scene_corners, Scalar(0, 0, 0));
            // Display the resulting frame
            imshow("Video Projection", img_output );
            moveWindow("Video Projection", img_object_col+img_output.cols,0);

            // Press  ESC on keyboard to exit
            char c=(char)waitKey(25);
            if(c==27)
            break;
        }
        cap.release();
    }

  
    waitKey();
    return 0;
}
