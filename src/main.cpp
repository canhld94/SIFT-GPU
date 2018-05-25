#include "sift.hpp"

using namespace cv;
using namespace cv::xfeatures2d;

void readme();

/** @function main */
int main( int argc, char** argv )
{
  if( argc != 2 )
  { readme(); return -1; }

  Mat img, img1; // image
  std::vector<KeyPoint> keypoints; // keypoints to store keypoints
  Mat descriptors; // image descriptors
  double t, tf;
  img = imread( argv[1], IMREAD_GRAYSCALE );
  // resize(img, img,Size(640, 480), 0, 0, INTER_NEAREST);
  // GaussianBlur(img, img, Size(), 1.6,1.6);
  img.convertTo(img1, CV_32F);
  if(!img.data)
  { std::cout<< " --(!) Error reading images " << std::endl; return -1; }
// SITF_BuildIn_OpenCV(img, keypoints, descriptors);
   std::vector<Mat> gpyr, dogpyr;
   tf = getTickFrequency();
   t = (double) getTickCount();
   buildGaussianPyramid(img1, gpyr, 5);
   buildDoGPyramid(gpyr, dogpyr, 5);
   t = (double) getTickCount() - t;
   printf("pyramid construction time: %g\n", t*1000./tf);
//    char s[100];
//    for (int i = 0; i < gpyr.size(); ++i) {
//  	sprintf(s, "DoGaussian %d", i);
//  	normalize(gpyr[i], gpyr[i], 0, 1, NORM_MINMAX);
//  	imshow(s, gpyr[i]);
//  }
  t = (double) getTickCount();
  findScaleSpaceExtrema(gpyr,dogpyr,keypoints, 5);
  t = (double) getTickCount() - t;
   printf("keypoint localization time: %g\n", t*1000./tf);
  Mat img_keypoints;
  drawKeypoints( img, keypoints, img_keypoints, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
  imshow("Keypoints", img_keypoints);
  waitKey(0);

  return 0;
  }

  /** @function readme */
  void readme(){
	  std::cout<< "Usage: ./SIFT_detector <img1>" << std::endl;
  }
