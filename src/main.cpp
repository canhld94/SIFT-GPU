#include "sift.hpp"

using namespace cv;
using namespace cv::xfeatures2d;

void readme();

/** @function main */
int main( int argc, char** argv )
{
  if( argc != 2 )
  { readme(); return -1; }

  Mat img; // image
  std::vector<KeyPoint> keypoints; // keypoints to store keypoints
  Mat descriptors; // image descriptors
  double t, tf;
  img = imread( argv[1], IMREAD_GRAYSCALE );
  img.convertTo(img, CV_32F);
  if(!img.data)
  { std::cout<< " --(!) Error reading images " << std::endl; return -1; }
// resize(img, img,Size(320, 240), 0, 0, INTER_NEAREST);
// SITF_BuildIn_OpenCV(img, keypoints, descriptors);
   std::vector<Mat> gpyr, dogpyr;
   tf = getTickFrequency();
   t = (double) getTickCount();
   buildGaussianPyramid(img, gpyr, 5);
   buildDoGPyramid(gpyr, dogpyr, 5);
   t = (double) getTickCount() - t;
   printf("pyramid construction time: %g\n", t*1000./tf);   char s[100];
   for (int i = 0; i < dogpyr.size(); ++i) {
 	sprintf(s, "DoGaussian %d", i);
 	normalize(dogpyr[i], dogpyr[i], 0, 1, NORM_MINMAX);
 	imshow(s, dogpyr[i]);
 }
  waitKey(0);

  return 0;
  }

  /** @function readme */
  void readme(){
	  std::cout<< "Usage: ./SIFT_detector <img1>" << std::endl;
  }
