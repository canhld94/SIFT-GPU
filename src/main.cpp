#include "sift.hpp"

using namespace cv;
using namespace cv::xfeatures2d;

void readme();
void readImage(Mat&, Mat&, char*, bool);

/** @function main */
int main( int argc, char** argv )
{
    if( argc != 3 )
      { readme(); return -1; }

    Mat img0, img1; // image
    Mat gray0, gray1;
    std::vector<KeyPoint> keypoints0, keypoints1; // keypoints to store keypoints
    Mat descriptors0, descriptors1; // image descriptor
    readImage(img0, gray0 , argv[1], 1);
    readImage(img1, gray1, argv[2], 0);
    // SITF_BuildIn_OpenCV(gray0, keypoints0, descriptors0);
    // SITF_BuildIn_OpenCV(gray1, keypoints1, descriptors1);
    SIFT_NCL(gray0, keypoints0, descriptors0);
    SIFT_NCL(gray1, keypoints1, descriptors1);
    BFMatcher matcher(NORM_L1);
    std::vector<std::vector<DMatch> > matches;
    matcher.knnMatch(descriptors1, descriptors0, matches, 2);
    std::vector<DMatch> good_matches;
    good_matches.reserve(matches.size());  
    for (size_t i = 0; i < matches.size(); ++i)
    { 
        if (matches[i].size() < 2)
                    continue;
      
        const DMatch &m1 = matches[i][0];
        const DMatch &m2 = matches[i][1];
            
        if(m1.distance <= 0.85*m2.distance)        
        good_matches.push_back(m1);     
  }
    Mat img_matches;
    drawMatches(img1, keypoints1, img0, keypoints0, good_matches, img_matches, Scalar(0, 0, 255), Scalar::all(-1),std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    
    std::vector< Point2f >  obj;
    std::vector< Point2f >  scene;
 
    for( unsigned int i = 0; i < good_matches.size(); i++ )
    {
        //-- Get the keypoints from the good matches
        obj.push_back( keypoints1[ good_matches[i].queryIdx ].pt );
        scene.push_back( keypoints0[ good_matches[i].trainIdx ].pt );
    }
 
    Mat H = findHomography( obj, scene, RANSAC );

    //-- Get the corners from the image_1 ( the object to be "detected" )
    std::vector< Point2f > obj_corners(4);
    obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( img1.cols, 0 );
    obj_corners[2] = cvPoint( img1.cols, img1.rows ); obj_corners[3] = cvPoint( 0, img1.rows );
    std::vector< Point2f > scene_corners(4);
 
    perspectiveTransform( obj_corners, scene_corners, H);
 
//-- Draw lines between the corners (the mapped object in the scene - image_2 ) 
  line( img_matches, scene_corners[0] + Point2f( img1.cols, 0), scene_corners[1] + Point2f( img1.cols, 0), Scalar(0, 255, 0), 4 );
  line( img_matches, scene_corners[1] + Point2f( img1.cols, 0), scene_corners[2] + Point2f( img1.cols, 0), Scalar( 0, 255, 0), 4 );
  line( img_matches, scene_corners[2] + Point2f( img1.cols, 0), scene_corners[3] + Point2f( img1.cols, 0), Scalar( 0, 255, 0), 4 );
  line( img_matches, scene_corners[3] + Point2f( img1.cols, 0), scene_corners[0] + Point2f( img1.cols, 0), Scalar( 0, 255, 0), 4 );
    imshow("Keypoints", img_matches);
    waitKey(0);
    return 0;
}

  /** @function readme */
  void readme(){
	  std::cout<< "Usage: ./sift <scene> <object> " << std::endl;
  }
  
  void readImage(Mat& img, Mat& gray, char* filename, bool resized){
    img = imread(filename);
    if(!img.data)
      { std::cout<< " --(!) Error reading images " << std::endl; exit(0); }
    if(resized) resize(img, img, Size(800,800));
    cvtColor(img, gray, cv::COLOR_RGB2GRAY);
    gray.convertTo(gray, CV_32F);
    return;
  }
