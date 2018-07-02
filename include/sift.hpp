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

#define DATATYPE CV_32FC1

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
