#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int main( int argc, char* argv[] )
{
    VideoCapture cap("ar_source.mov"); 
   
  // Check if camera opened successfully
  if(!cap.isOpened()){
    cout << "Error opening video stream or file" << endl;
    return -1;
  }
	
  while(1){

    Mat frame;
    // Capture frame-by-frame
    cap >> frame;
 
    // If the frame is empty, break immediately 
    if (frame.empty())
      break;

    Rect ROI(0, 50, 640, 260);
    Mat cropped_frame = frame(ROI);
    // resize(frame, frame, Size(320, 360), 0, 0, INTER_CUBIC);
    // Display the resulting frame
    imshow( "Frame", cropped_frame );

    // Press  ESC on keyboard to exit
    char c=(char)waitKey(25);
    if(c==27)
      break;
  }
 
  // When everything done, release the video capture object
  cap.release();

  // Closes all the frames
//   destroyAllWindows();

    waitKey();
    return 0;
}