# include "opencv2/core/core.hpp"
# include "opencv2/features2d/features2d.hpp"
# include "opencv2/highgui/highgui.hpp"
# include "opencv2/calib3d/calib3d.hpp"
# include "opencv2/nonfree/features2d.hpp"

#include <iostream>
#include <vector>

using namespace cv;

void readme();

/**
 * @function main
 * @brief Main function
 */
int main( int argc, char** argv )
{
  if( argc != 3 )
  { readme(); return -1; }

  Mat img_object = imread( argv[1], CV_LOAD_IMAGE_GRAYSCALE );
  Mat img_scene = imread( argv[2], CV_LOAD_IMAGE_GRAYSCALE );

  if( !img_object.data || !img_scene.data )
  { printf(" --(!) Error reading images \n"); return -1; }

  //-- Step 1: Detect the keypoints using SURF Detector
  int minHessian = 400;

  SurfFeatureDetector detector( minHessian );

  std::vector<KeyPoint> keypoints_object, keypoints_scene;

  detector.detect( img_object, keypoints_object );
  detector.detect( img_scene, keypoints_scene );

  //-- Step 2: Calculate descriptors (feature vectors)
  SurfDescriptorExtractor extractor;

  Mat descriptors_object, descriptors_scene;

  extractor.compute( img_object, keypoints_object, descriptors_object );
  extractor.compute( img_scene, keypoints_scene, descriptors_scene );

  //-- Step 3: Matching descriptor vectors using FLANN matcher
  FlannBasedMatcher matcher;
  std::vector< DMatch > matches;
  matcher.match( descriptors_object, descriptors_scene, matches );

  double max_dist = 0; double min_dist = 100;

  //-- Quick calculation of max and min distances between keypoints
  for( int i = 0; i < descriptors_object.rows; i++ )
  { double dist = matches[i].distance;
    if( dist < min_dist ) min_dist = dist;
    if( dist > max_dist ) max_dist = dist;
  }

  printf("-- Max dist : %f \n", max_dist );
  printf("-- Min dist : %f \n", min_dist );

  //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
  std::vector< DMatch > good_matches;

  for( int i = 0; i < descriptors_object.rows; i++ )
  { if( matches[i].distance < 3*min_dist )
    { good_matches.push_back( matches[i]); }
  }

  Mat img_matches;
  drawMatches( img_object, keypoints_object, img_scene, keypoints_scene,
               good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );


  //-- Localize the object from img_1 in img_2
  std::vector<Point2f> obj;
  std::vector<Point2f> scene;

  for( size_t i = 0; i < good_matches.size(); i++ )
  {
    //-- Get the keypoints from the good matches
    obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
    scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
  }

  Mat K = (Mat_<double>(3,3) << 800, 0, 320, 0, 800, 240, 0, 0, 1);

  Mat F = findFundamentalMat(obj, scene);

  Mat w;
  SVD::compute(F, w);

  std::cout << "F\n" << F << std::endl;
  std::cout << w << std::endl;

  Mat E = K.t() * F * K;
  Mat u, vt;
  SVD::compute(E, w, u, vt);

  std::cout << "BEFORE:\n";
  std::cout << "E\n" << E << std::endl;
  std::cout << "u\n" << u << std::endl;
  std::cout << "w\n" << w << std::endl;
  std::cout << "vt\n" << vt << std::endl;

  w.at<double>(0,0) = ( w.at<double>(0,0) + w.at<double>(1,0) ) / 2.0;
  w.at<double>(1,0) = w.at<double>(0,0);
  w.at<double>(2,0) = 0.0;
  Mat w_ = (Mat_<double>(3,3) << w.at<double>(0,0), 0, 0, 0, w.at<double>(1,0), 0, 0, 0, w.at<double>(2,0));

  std::cout << "BEFORE_2:\n";
  std::cout << "E\n" << E << std::endl;
  std::cout << "u\n" << u << std::endl;
  std::cout << "w\n" << w_ << std::endl;
  std::cout << "vt\n" << vt << std::endl;

  E = u * w_ * vt;
  SVD::compute(E, w, u, vt);
  w_ = (Mat_<double>(3,3) << w.at<double>(0,0), 0, 0, 0, w.at<double>(1,0), 0, 0, 0, w.at<double>(2,0));
  std::cout << "AFTER:\n";
  std::cout << "E\n" << E << std::endl;
  std::cout << "u\n" << u << std::endl;
  std::cout << "w\n" << w_ << std::endl;
  std::cout << "vt\n" << vt << std::endl;

  //extract all four possible translations and rotations
  vector<Mat> p_r, p_t;
  p_r.push_back(u*w_*vt);     p_t.push_back(u.col(2));
  p_r.push_back(u*w_*vt);     p_t.push_back(u.col(2));
  p_r.push_back(u*w_.t()*vt); p_t.push_back(-1*u.col(2));
  p_r.push_back(u*w_.t()*vt); p_t.push_back(-1*u.col(2));

  for(int i = 0; i < 4; i++)
  {
    Mat t1 = (Mat_<double>(3,1) << 0, 0, 0); Mat R1 = Mat::eye(3, 3, CV_64F);
    Mat t2 = p_t[i]; Mat R2 = p_r[i];
    Mat T1, T2;
    hconcat(R1, t1, T1); 
    hconcat(R2, t2, T2);
    Mat M1 = K*T1;
    Mat M2 = K*T2;
    cv::Mat pt(1,4,CV_64FC4);
    cv::Mat cam0pts(1,3,CV_64FC2);
    cam0pts.at<float>(0,0) = obj[0].x;
    cam0pts.at<float>(0,1) = obj[0].y;
    cam0pts.at<float>(0,2) = 1;
    cv::Mat cam1pts(1,3,CV_64FC2);
    cam1pts.at<float>(0,0) = scene[0].x;
    cam1pts.at<float>(0,1) = scene[0].y;
    cam1pts.at<float>(0,2) = 1;
    triangulatePoints(M1, M2, cam0pts, cam1pts, pt);

    std::cout << "R1\n" << R1 << "\nR2\n" << R2 << std::endl; 

    std::cout << "X_1: " << pt.at<float>(0,0)/pt.at<float>(0,3) << ", ";
    std::cout << "Y_1: " << pt.at<float>(0,1)/pt.at<float>(0,3) << ", ";
    std::cout << "Z_1: " << pt.at<float>(0,2)/pt.at<float>(0,3) << ", ";
    std::cout << "1_1: " << pt.at<float>(0,3)/pt.at<float>(0,3) << std::endl;

    Mat pt_2 = T2 * pt;
    std::cout << "X_2: " << pt_2.at<float>(0,0)/pt_2.at<float>(0,3) << ", ";
    std::cout << "Y_2: " << pt_2.at<float>(0,1)/pt_2.at<float>(0,3) << ", ";
    std::cout << "Z_2: " << pt_2.at<float>(0,2)/pt_2.at<float>(0,3) << ", ";
    std::cout << "1_2: " << pt_2.at<float>(0,3)/pt_2.at<float>(0,3) << std::endl << std::endl;
  }
  //-- Get the corners from the image_1 ( the object to be "detected" )
  std::vector<Point2f> obj_corners(4);
  obj_corners[0] = Point(0,0); obj_corners[1] = Point( img_object.cols, 0 );
  obj_corners[2] = Point( img_object.cols, img_object.rows ); obj_corners[3] = Point( 0, img_object.rows );
  std::vector<Point2f> scene_corners(4);

  perspectiveTransform( obj_corners, scene_corners, F );


  //-- Draw lines between the corners (the mapped object in the scene - image_2 )
  Point2f offset( (float)img_object.cols, 0);
  line( img_matches, scene_corners[0] + offset, scene_corners[1] + offset, Scalar(0, 255, 0), 4 );
  line( img_matches, scene_corners[1] + offset, scene_corners[2] + offset, Scalar( 0, 255, 0), 4 );
  line( img_matches, scene_corners[2] + offset, scene_corners[3] + offset, Scalar( 0, 255, 0), 4 );
  line( img_matches, scene_corners[3] + offset, scene_corners[0] + offset, Scalar( 0, 255, 0), 4 );

  //-- Show detected matches
  imshow( "Good Matches & Object detection", img_matches );

  waitKey(0);

  return 0;
}

/**
 * @function readme
 */
void readme()
{ printf(" Usage: ./SURF_Homography <img1> <img2>\n"); }