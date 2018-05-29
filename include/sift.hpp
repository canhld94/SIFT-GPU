/*
 * sift.hpp
 *
 *  Created on: May 18, 2018
 *      Author: canhld
 */

#ifndef SIFT_HPP_
#define SIFT_HPP_

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/core/hal/hal.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/types_c.h>
#include <iostream>
#include <sys/time.h>
#include <omp.h>

using namespace cv;
using namespace cv::xfeatures2d;

typedef float data_t; // data type using in filter

/* Parameter to implement SIFT*/
static const int nOctaveLayers = 2; // number of layer per octaves
static const int nOctaves = 5;
static const int nScales = 5; // nOctaveLayers + 3
static const double Sigma = 1.6; // initial sigma
static const double PI = 3.14159265359;
static const double contrastThreshold = 0.04;
static const double edgeThreshold = 10;

// default width of descriptor histogram array
static const int SIFT_DESCR_WIDTH = 4;

// default number of bins per histogram in descriptor array
static const int SIFT_DESCR_HIST_BINS = 8;

// assumed gaussian blur for input image
static const float SIFT_INIT_SIGMA = 0.5f;

// width of border in which to ignore keypoints
static const int SIFT_IMG_BORDER = 5;

// maximum steps of keypoint interpolation before failure
static const int SIFT_MAX_INTERP_STEPS = 5;

// default number of bins in histogram for orientation assignment
static const int SIFT_ORI_HIST_BINS = 36;

// determines gaussian sigma for orientation assignment
static const float SIFT_ORI_SIG_FCTR = 1.5f;

// determines the radius of the region used in orientation assignment
static const float SIFT_ORI_RADIUS = 3 * SIFT_ORI_SIG_FCTR;

// orientation magnitude relative to max that results in new feature
static const float SIFT_ORI_PEAK_RATIO = 0.8f;

// determines the size of a single descriptor orientation histogram
static const float SIFT_DESCR_SCL_FCTR = 3.f;

// threshold on magnitude of elements of descriptor vector
static const float SIFT_DESCR_MAG_THR = 0.2f;

// factor used to convert floating-point descriptor to unsigned char
static const float SIFT_INT_DESCR_FCTR = 512.f;

static const int SIFT_FIXPT_SCALE = 1;

 /*SIFT build-in opencv function*/
void SITF_BuildIn_OpenCV(InputArray image,
						 std::vector<KeyPoint>& keypoints,
						 OutputArray descriptors);

 /*NCL SIFT, based opencv source code*/
void SIFT_NCL(InputArray image,
		  std::vector<KeyPoint> & keypoints,
		  OutputArray descriptors);

/*Sub modules*/

void Gaussian_Blur(Mat& src, Mat& dst, double sigma);

void Gaussian_Blur_1D(Mat& src, Mat& dst, double sigma);

void buildGaussianPyramid(Mat& image,
						std::vector<Mat>& gpyr,
						int nOctaves);

void buildDoGPyramid(std::vector<Mat>& gpyr,
					 std::vector<Mat>& dogpyr,
					 int nOctaves);

void findScaleSpaceExtrema(std::vector<Mat>& gpyr,
						   std::vector<Mat>& dogpyr,
						   std::vector<KeyPoint>& keypoints,
						   int nOctaves);

void calDescriptor( std::vector<Mat>& gpyr,
					std::vector<KeyPoint>& keypoints,
					Mat& descriptors,
					int firstOctave);


#endif /* SIFT_HPP_ */
