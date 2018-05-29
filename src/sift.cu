#include "sift.hpp"
#include "sift_cuda.hpp"
#include "cuda_helper.h"

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX_THREADS_PER_BLOCK 1024

__global__ void gaussianBlurRow(float* input, float* output, float* filter, int rows, int cols, int filter_size){
    int tid_row = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_col = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = tid_row * cols + tid_col;
    if(tid_row >= rows || tid_col >= cols)
        return;
    float tmp = 0;
    int half_size = filter_size/2;
    for(int i = -half_size ; i < half_size ; i++){
        tmp += (tid_col + i < 0 || tid_col + i >= cols - 1 ) ? 0 : input[tid + i] * filter[i + half_size];
    }
    output[tid] = tmp;
}
__global__ void gaussianBlurCol(float* input, float* output, float* filter, int rows, int cols, int filter_size){
    int tid_row = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_col = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = tid_row * cols + tid_col;
    if(tid_row >= rows || tid_col >= cols)
        return;
    float tmp = 0;
    int half_size = filter_size/2;
    for(int i = -half_size ; i < half_size ; i++){
        tmp += (tid_row + i < 0 || tid_row + i >= rows - 1 ) ? 0 : input[ (tid_row + i) * cols + tid_col ] * filter[i + half_size];
    }
    output[tid] = tmp;
}

__global__ void gaussianFilter1D(float* kernel_data,float sigma,int w){
    int tid = threadIdx.x;
    double dat = 1. / sqrt(2 * PI * sigma * sigma) * exp(-((double) (tid - w) * (tid - w)) * 1. / (2 * sigma * sigma));
    kernel_data[tid] = (float)dat;
}

__global__ void halfImage(float* input, float* output, int rows, int cols){
    int tid_row = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_col = blockIdx.y * blockDim.y + threadIdx.y;

    int dst_rows = rows/2;
    int dst_cols = cols/2;

    if( tid_col * 2 > cols || tid_row * 2 > rows)
        return;

    output[ tid_row * dst_cols + tid_col ] = input[ (tid_row*2) * cols + (tid_col*2) ];
}

__global__ void differentiate(float* input1, float* input2, float* output, int rows, int cols){
    int tid_row = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_col = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = tid_row * cols + tid_col;
    if(tid >= rows*cols)
        return;
    output[tid] = input2[tid] - input1[tid];
}

extern "C" void getGaussianKernel1DGPU(float sigma, float* d_kernel_data, int w){
    int size = 2 * w + 1;
    dim3 block(size);
    gaussianFilter1D<<<1,block>>>(d_kernel_data,sigma,w);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
}

extern "C" void gaussianBlur1DGPU(float* src, float* dst, float* filter, int rows, int cols, int filter_size) {
    float *itm_data;

    CHECK(cudaMalloc((float**)&itm_data, rows*cols * sizeof(float)));

    dim3 gridRow(rows/32 + (rows%32?1:0),cols/32 + (cols%32?1:0));
    dim3 blockRow(MIN(rows,32),MIN(cols,32));
    gaussianBlurRow<<<gridRow,blockRow>>>(src,itm_data,filter,rows,cols,filter_size);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    dim3 gridCol(rows/32 + (rows%32?1:0),cols/32 + (cols%32?1:0));
    dim3 blockCol(MIN(rows,32),MIN(cols,32));
    gaussianBlurCol<<<gridCol,blockCol>>>(itm_data,dst,filter,rows,cols,filter_size);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaFree(itm_data));
}

extern "C" void halfImageGPU(float* src, float* dst, int rows_ori, int cols_ori){
    int rows = rows_ori / 2;
    int cols = cols_ori / 2;
    dim3 grid(rows/32 + (rows%32?1:0),cols/32 + (cols%32?1:0));
    dim3 block(MIN(rows,32),MIN(cols,32));
    halfImage<<<grid,block>>>(src,dst,rows_ori,cols_ori);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
}

extern "C" void differentiateGPU(float* src1, float* src2, float* dst, int rows, int cols){
    dim3 grid(rows/32 + (rows%32?1:0),cols/32 + (cols%32?1:0));
    dim3 block(MIN(rows,32),MIN(cols,32));
    differentiate<<<grid,block>>>(src1, src2, dst, rows, cols);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
}

extern "C" void SIFT_NCL_GPU(InputArray image,
    std::vector<KeyPoint> & keypoints,
    OutputArray descriptors){
        std::vector<Mat> dogpyr(nOctaves*(nScales - 1));
        std::vector<Mat> gpyr(nOctaves*nScales);
        double t, tf;
        tf = getTickFrequency();
        t = (double) getTickCount();
        float** gpyr_arr = new float*[nOctaves*nScales];
        float** dogpyr_arr = new float*[nOctaves*(nScales-1)];
        float** filter_arr = new float*[nScales];
        std::vector<int> filter_sizes(nScales);
        std::vector<int> rows(nOctaves);
        std::vector<int> cols(nOctaves);
        Mat img = image.getMat();
        prepareForGPU(img, filter_arr, gpyr_arr, dogpyr_arr, rows, cols, filter_sizes);
        for(int i = 0 ; i < nOctaves ; i++){
            buildGaussianPyramidGPU(i, img, filter_arr, gpyr_arr, rows, cols, filter_sizes);
            buildDoGPyramidGPU(i, gpyr_arr, dogpyr_arr, rows, cols);
        }
        prepareForCPU(gpyr_arr, dogpyr_arr, gpyr, dogpyr, rows, cols);
        t = (double) getTickCount() - t;
        printf("pyramid construction time: %g\n", t*1000./tf);
        t = (double) getTickCount();
        findScaleSpaceExtrema(gpyr,dogpyr,keypoints, 5);
        t = (double) getTickCount() - t;
        printf("keypoint localization time: %g\n", t*1000./tf);

        t = (double) getTickCount();
        int dsize = SIFT_DESCR_WIDTH*SIFT_DESCR_WIDTH*SIFT_DESCR_HIST_BINS;
        descriptors.create((int)keypoints.size(), dsize, CV_32F);
        Mat _descriptors = descriptors.getMat();
        calDescriptor(gpyr, keypoints, _descriptors, 0);
        t = (double) getTickCount() - t;
        printf("descriptor extraction time: %g\n", t*1000./tf);

        return;
}

extern "C" void prepareForGPU(Mat img_mat, float** filter, float** gpyr, float** dogpyr, std::vector<int>& rows, std::vector<int>& cols, std::vector<int>& filter_sizes){
	std::vector<float> sig(nScales);
    float *img_arr = (float *) img_mat.data;
    int row_ori = img_mat.rows;
    int col_ori = img_mat.cols;

    double k = pow(2.0, 1.0/nOctaveLayers);
    for(int i = 0 ; i < nOctaves ; i++){
        rows[i] = row_ori;
        cols[i] = col_ori;
        row_ori = floor(row_ori/2);
        col_ori = floor(col_ori/2);
        for(int j = 0 ; j < nScales ; j++){
            CHECK(cudaMalloc((float**)&(gpyr[i*nScales + j]), rows[i] * cols[i] * sizeof(float) ));
        }
        for(int j = 0 ; j < nScales - 1 ; j++){
            CHECK(cudaMalloc((float**)&(dogpyr[i*(nScales - 1) + j]), rows[i] * cols[i] * sizeof(float) ));
        }
    }
	for(int i = 0; i < nScales; i++){
        if(i == 0){
            sig[i] = Sigma;
        }else{
            double sig_total = pow(k* 1.0, (double) i)*Sigma;
            sig[i] = (float) sqrt(sig_total*sig_total - Sigma*Sigma);
        }

        int w = floor(3 * sig[i]);
        int size = 2 * w + 1;
        filter_sizes[i] = size;

        CHECK(cudaMalloc((float**)&filter[i], size * sizeof(float)));
        getGaussianKernel1DGPU(sig[i],filter[i],w);
	}
    CHECK(cudaMemcpy(gpyr[0],img_arr,rows[0] * cols[0] * sizeof(float),cudaMemcpyHostToDevice));
    
    double init_sigma = sqrt(Sigma*Sigma + 0.5*0.5);
    float* init_kernel;
    int init_w = floor(3 * init_sigma);
    int init_ksize = 2 * init_w + 1;

    CHECK(cudaMalloc((float**)&init_kernel, init_ksize * sizeof(float)));
    getGaussianKernel1DGPU(init_sigma, init_kernel, init_w);
    gaussianBlur1DGPU(gpyr[0],gpyr[0],init_kernel,rows[0],cols[0],init_ksize);
}

extern "C" void prepareForCPU(float** gpyr_d, float** dogpyr_d, std::vector<Mat>& gpyr, std::vector<Mat>& dogpyr, std::vector<int>& rows, std::vector<int>& cols){
    for(int i = 0 ; i < nOctaves ; i++){
        for(int j = 0 ; j < nScales ; j++){
            int id = i*nScales + j;
            gpyr[id] = cv::Mat(rows[i], cols[i], CV_32F);
            float* gpyr_h = (float*) gpyr[id].data;
            CHECK(cudaMemcpy(gpyr_h,gpyr_d[id], rows[i] * cols[i] * sizeof(float),cudaMemcpyDeviceToHost));
        }
        for(int j = 0 ; j < (nScales - 1) ; j++){
            int id = i*(nScales-1) + j;
            dogpyr[id] = cv::Mat(rows[i], cols[i], CV_32F);
            float* dogpyr_h = (float*) dogpyr[id].data;
            CHECK(cudaMemcpy(dogpyr_h,dogpyr_d[id], rows[i] * cols[i] * sizeof(float),cudaMemcpyDeviceToHost));
        }
    }
}

extern "C" void buildGaussianPyramidGPU(int octave_idx, Mat image, float** filter, float** gpyr, std::vector<int>& rows, std::vector<int>& cols, std::vector<int>& filter_sizes){
    for(int j = 0 ; j < nScales ; j++ ){
        if(octave_idx == 0 && j == 0){
        }else if(j == 0){
            halfImageGPU(gpyr[(octave_idx-1)*nScales + nOctaveLayers],gpyr[octave_idx*nScales],rows[octave_idx-1],cols[octave_idx-1]);
        }else{
            gaussianBlur1DGPU(gpyr[octave_idx*nScales],gpyr[octave_idx*nScales+j],filter[j],rows[octave_idx],cols[octave_idx],filter_sizes[j]);
        }
    }
}
extern "C" void buildDoGPyramidGPU(int octave_idx, float** gpyr, float** dogpyr, std::vector<int>& rows, std::vector<int>& cols){
    for(int j = 0 ; j < nScales - 1 ; j++ ){
        differentiateGPU(gpyr[octave_idx*nScales+j],gpyr[octave_idx*nScales+j+1],dogpyr[octave_idx*(nScales - 1) + j], rows[octave_idx], cols[octave_idx]);
    }
}