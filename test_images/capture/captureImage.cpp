#include "opencv2/opencv.hpp"
# include "opencv2/nonfree/features2d.hpp"
#include <string>

using namespace cv;

int main(int, char** argv)
{
    VideoCapture cap(std::stoi(std::string(argv[1]))); // open the default camera
    if(!cap.isOpened())  // check if we succeeded
        return -1;

    namedWindow("capture",1);
    Mat frame_small, frame_small_gray;
    int i = 1;
    for(;;)
    {
        Mat frame;
        cap >> frame; // get a new frame from camera
        
        resize(frame, frame_small, Size(frame.cols/2,frame.rows/2));
        cvtColor(frame_small, frame_small_gray, cv::COLOR_RGB2GRAY);

        //-- Step 1: Detect the keypoints using SURF Detector
          int minHessian = 400;

          SurfFeatureDetector detector( minHessian );

          std::vector<KeyPoint> keypoints_1;

          detector.detect( frame_small_gray, keypoints_1 );

          //-- Draw keypoints
          Mat img_keypoints_1;

          drawKeypoints( frame_small_gray, keypoints_1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );

          //-- Show detected (drawn) keypoints
          imshow("capture", img_keypoints_1 );
        if(waitKey(30) >= 0)
        {
            imwrite(std::string("capture")+std::to_string(i)+".jpg", frame_small_gray);
            i++;
        }
        else if( i > 2 )
        {
            break;
        }
    }
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}