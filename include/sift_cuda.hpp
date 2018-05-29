#ifndef _CUDA_H
#define _CUDA_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <stdlib.h>
#include <vector>

__global__ void gaussianBlurRow(float* input, float* output, float* filter, int rows, int cols, int filter_size);
__global__ void gaussianBlurCol(float* input, float* output, float* filter, int rows, int cols, int filter_size);
__global__ void gaussianFilter1D(float* kernel_data,float sigma,int w);
__global__ void halfImage(float* input, float* output, int rows, int cols);
__global__ void differentiate(float* input1, float* input2, float* output, int rows, int cols);

extern "C" void SIFT_NCL_GPU(InputArray image,std::vector<KeyPoint> & keypoints,OutputArray descriptors);

extern "C" void getGaussianKernel1DGPU(float sigma, float* d_kernel_data, int w);
extern "C" void gaussianBlur1DGPU(float* src, float* dst, float* filter, int rows, int cols, int filter_size);
extern "C" void halfImageGPU(float* src, float* dst, int rows_ori, int cols_ori);
extern "C" void differentiateGPU(float* src1, float* src2, float* dst, int rows, int cols);
extern "C" void prepareForGPU(Mat img_mat, float** filter, float** gpyr, float** dogpyr, std::vector<int>& rows, std::vector<int>& cols, std::vector<int>& filter_sizes);
extern "C" void prepareForCPU(float** gpyr_arr, float** dogpyr_d, std::vector<Mat>& gpyr, std::vector<Mat>& dogpyr, std::vector<int>& rows, std::vector<int>& cols);
extern "C" void buildGaussianPyramidGPU(int octave_idx, Mat image, float** filter, float** gpyr, std::vector<int>& rows, std::vector<int>& cols, std::vector<int>& filter_sizes);
extern "C" void buildDoGPyramidGPU(int octave_idx, float** gpyr, float** dogpyr, std::vector<int>& rows, std::vector<int>& cols);

#endif