#include "sift.hpp"
#include "sift_cuda.hpp"
#include "cuda_helper.h"

#define MAX_THREADS_PER_BLOCK 1024

__global__ void gaussianBlurRow(float* input_g, float* output, float* filter, int rows, int cols, int filter_size){
    extern __shared__ float shared_ptr[];
    float* filter_s = shared_ptr;
    float* input_s = (float*)&shared_ptr[filter_size];

    int half_size = filter_size/2;
    int row_idx = blockIdx.x;
    int col_idx = blockIdx.y * blockDim.x + threadIdx.x;
    int input_idx = row_idx * cols + col_idx;

    // Filter to shared memory
    if(threadIdx.x < filter_size){
        filter_s[threadIdx.x] = filter[threadIdx.x];
    }
    if(blockDim.x < filter_size && threadIdx.x == 0){
        for(int j = blockDim.x ; j < filter_size ; j++){
            filter_s[j] = filter[j];
        }
    }

    // Input image to shared memory
    int row_offset = row_idx * cols - half_size;
    if(threadIdx.x < cols + half_size*2){
        if(threadIdx.x < half_size || threadIdx.x >= cols + half_size){
            input_s[threadIdx.x] = 0;
        }else{
            input_s[threadIdx.x] = input_g[row_offset + threadIdx.x];
        }
    }
    if(blockDim.x < cols + half_size*2 && threadIdx.x == 0){
        for(int j = blockDim.x ; j < cols + half_size*2 ; j++){
            if(j < half_size || j >= cols + half_size){
                input_s[j] = 0;
            }else{
                input_s[j] = input_g[row_offset + j];
            }
        }
    }
    __syncthreads();
    if(col_idx >= cols)
        return;
    float tmp = 0;
    for(int i = -half_size ; i < half_size ; i++){
        tmp += input_s[col_idx + i + half_size] * filter_s[i + half_size];
    }
    output[input_idx] = tmp;
}
__global__ void gaussianBlurCol(float* input_g, float* output, float* filter, int rows, int cols, int filter_size){
    extern __shared__ float shared_ptr[];
    float* filter_s = shared_ptr;
    float* input_s = (float*)&shared_ptr[filter_size];

    int half_size = filter_size/2;
    int row_idx = blockIdx.y * blockDim.x + threadIdx.x;
    int col_idx = blockIdx.x;
    int input_idx = row_idx * cols + col_idx;

    // Filter to shared memory
    if(threadIdx.x < filter_size){
        filter_s[threadIdx.x] = filter[threadIdx.x];
    }
    if(blockDim.x < filter_size && threadIdx.x == 0){
        for(int j = blockDim.x ; j < filter_size ; j++){
            filter_s[j] = filter[j];
        }
    }

    // Input image to shared memory
    if(threadIdx.x < rows + half_size*2){
        if(threadIdx.x < half_size || threadIdx.x >= rows + half_size){
            input_s[threadIdx.x] = 0;
        }else{
            input_s[threadIdx.x] = input_g[(threadIdx.x - half_size)*cols + col_idx];
        }
    }
    if(blockDim.x < rows + half_size*2 && threadIdx.x == 0){
        for(int j = blockDim.x ; j < rows + half_size*2 ; j++){
            if(j < half_size || j >= rows + half_size){
                input_s[j] = 0;
            }else{
                input_s[j] = input_g[(j - half_size)*cols + col_idx];
            }
        }
    }
    __syncthreads();
    if(row_idx >= rows)
        return;
    float tmp = 0;
    for(int i = -half_size ; i < half_size ; i++){
        tmp += input_s[row_idx + i + half_size] * filter_s[i + half_size];
    }
    output[input_idx] = tmp;
}

__global__ void gaussianFilter1D(float* kernel_data,float sigma,int w){
    int tid = threadIdx.x;
    double dat = 1. / sqrt(2 * PI * sigma * sigma) * exp(-((double) (tid - w) * (tid - w)) * 1. / (2 * sigma * sigma));
    kernel_data[tid] = (float)dat;
}

__global__ void halfImage(float* input, float* output, int rows, int cols){
    int tid_row = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_col = blockIdx.y * blockDim.y + threadIdx.y;

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

extern "C" void gaussianBlur1DGPU(int octave_idx, int scale_idx, float* src, float* dst, float* dst_h, float* filter, float* itm_data, int rows, int cols, int filter_size, cudaStream_t stream) {
    //CHECK(cudaMalloc((float**)&itm_data, rows*cols * sizeof(float)));

    dim3 gridRow(rows,cols/MAX_THREADS_PER_BLOCK + (cols%MAX_THREADS_PER_BLOCK?1:0));
    dim3 blockRow(MIN(cols,MAX_THREADS_PER_BLOCK));
    gaussianBlurRow<<<gridRow,blockRow,( filter_size + (cols + (filter_size/2)*2) )*sizeof(float),stream>>>(src,itm_data,filter,rows,cols,filter_size);
    CHECK(cudaGetLastError());
    //CHECK(cudaDeviceSynchronize());
    CHECK(cudaStreamSynchronize(stream));

    dim3 gridCol(cols,rows/MAX_THREADS_PER_BLOCK + (rows%MAX_THREADS_PER_BLOCK?1:0));
    dim3 blockCol(MIN(rows,MAX_THREADS_PER_BLOCK));
    gaussianBlurCol<<<gridCol,blockCol,( filter_size + (rows + (filter_size/2)*2) )*sizeof(float),stream>>>(itm_data,dst,filter,rows,cols,filter_size);
    CHECK(cudaGetLastError());
    //CHECK(cudaDeviceSynchronize());
    //CHECK(cudaFree(itm_data));
    CHECK(cudaStreamSynchronize(stream));
    CHECK(cudaMemcpyAsync(dst_h,dst, rows * cols * sizeof(float),cudaMemcpyDeviceToHost,stream));
}

extern "C" void halfImageGPU(float* src, float* dst, float* dst_h, int rows_ori, int cols_ori, cudaStream_t stream){
    int rows = rows_ori / 2;
    int cols = cols_ori / 2;
    dim3 grid(rows/32 + (rows%32?1:0),cols/32 + (cols%32?1:0));
    dim3 block(MIN(rows,32),MIN(cols,32));
    halfImage<<<grid,block,0,stream>>>(src,dst,rows_ori,cols_ori);
    CHECK(cudaGetLastError());
    //CHECK(cudaDeviceSynchronize());
    CHECK(cudaStreamSynchronize(stream));
    CHECK(cudaMemcpyAsync(dst_h,dst, rows * cols * sizeof(float),cudaMemcpyDeviceToHost,stream));
}

extern "C" void differentiateGPU(float* src1, float* src2, float* dst, float* dst_h, int rows, int cols, cudaStream_t stream){
    dim3 grid(rows/32 + (rows%32?1:0),cols/32 + (cols%32?1:0));
    dim3 block(MIN(rows,32),MIN(cols,32));
    differentiate<<<grid,block,0,stream>>>(src1, src2, dst, rows, cols);
    CHECK(cudaGetLastError());
    //CHECK(cudaDeviceSynchronize());
    CHECK(cudaStreamSynchronize(stream));
    CHECK(cudaMemcpyAsync(dst_h,dst, rows * cols * sizeof(float),cudaMemcpyDeviceToHost,stream));
}

extern "C" void SIFT_NCL_GPU(InputArray image,
    std::vector<KeyPoint> & keypoints,
    OutputArray descriptors){
        std::vector<Mat> dogpyr(nOctaves*(nScales - 1));
        std::vector<Mat> gpyr(nOctaves*nScales);
        float* dummy;
        CHECK(cudaMalloc((float**)&dummy,1*sizeof(float)));
        CHECK(cudaFree(dummy));
        double t, tf;
        tf = getTickFrequency();
        t = (double) getTickCount();
        float** h_gpyr_arr = new float*[nOctaves*nScales];
        float** h_dogpyr_arr = new float*[nOctaves*(nScales-1)];
        float** gpyr_arr = new float*[nOctaves*nScales];
        float** dogpyr_arr = new float*[nOctaves*(nScales-1)];
        float** filter_arr = new float*[nScales];
        float** itm_arr = new float*[nScales];
        std::vector<int> filter_sizes(nScales);
        std::vector<int> rows(nOctaves);
        std::vector<int> cols(nOctaves);
        std::vector<cudaStream_t> streams(nScales);
        Mat img = image.getMat();
        prepareForGPU(img, filter_arr, gpyr_arr, h_gpyr_arr, gpyr, dogpyr_arr, h_dogpyr_arr, dogpyr, itm_arr, rows, cols, filter_sizes, streams);
        for(int i = 0 ; i < nOctaves ; i++){
            buildGaussianPyramidGPU(i, img, filter_arr, gpyr_arr, gpyr, itm_arr, rows, cols, filter_sizes,streams);
            buildDoGPyramidGPU(i, gpyr_arr, dogpyr_arr, dogpyr, rows, cols,streams);
        }
        prepareForCPU(gpyr_arr, dogpyr_arr, gpyr, dogpyr, itm_arr, rows, cols, streams);
        t = (double) getTickCount() - t;
        printf("pyramid construction time: %g\n", t*1000./tf);
        //char s[100];
	    //for (int i = 0; i < gpyr.size(); ++i) {
	    //    sprintf(s, "DoGaussian %d.png", i);
	    //    normalize(gpyr[i], gpyr[i], 0, 255, NORM_MINMAX);
	    //    imwrite(s, gpyr[i]);
	    //}
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
        CHECK(cudaDeviceReset());
        return;
}

extern "C" void prepareForGPU(Mat img_mat, float** filter, float** gpyr, float** h_gpyr,std::vector<Mat>& gpyr_mat, float** dogpyr, float** h_dogpyr, std::vector<Mat>& dogpyr_mat, float** itm_data, std::vector<int>& rows, std::vector<int>& cols, std::vector<int>& filter_sizes, std::vector<cudaStream_t>& streams){
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
            CHECK(cudaMallocHost((float**)&(h_gpyr[i*nScales + j]), rows[i] * cols[i] * sizeof(float)));
            gpyr_mat[i*nScales + j] = cv::Mat(rows[i], cols[i], CV_32F, h_gpyr[i*nScales + j], cols[i]*sizeof(float));
        }
        for(int j = 0 ; j < nScales - 1 ; j++){
            CHECK(cudaMalloc((float**)&(dogpyr[i*(nScales - 1) + j]), rows[i] * cols[i] * sizeof(float) ));
            CHECK(cudaMallocHost((float**)&(h_dogpyr[i*(nScales-1) + j]), rows[i] * cols[i] * sizeof(float)));
            dogpyr_mat[i*(nScales-1) + j] = cv::Mat(rows[i], cols[i], CV_32F, h_dogpyr[i*(nScales-1) + j],cols[i]*sizeof(float));
        }
    }
	for(int i = 0; i < nScales; i++){
        cudaStreamCreate(&(streams[i]));
        if(i == 0){
            sig[i] = Sigma;
        }else{
            double sig_total = pow(k* 1.0, (double) i)*Sigma;
            sig[i] = (float) sqrt(sig_total*sig_total - Sigma*Sigma);
        }

        int w = floor(3 * sig[i]);
        int size = 2 * w + 1;
        filter_sizes[i] = size;

        CHECK(cudaMalloc((float**)&itm_data[i],rows[0]*cols[0]*sizeof(float)));
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
    gaussianBlur1DGPU(0,0,gpyr[0],gpyr[0],(float*)gpyr_mat[0].data,init_kernel,itm_data[0],rows[0],cols[0],init_ksize,streams[0]);
}

extern "C" void prepareForCPU(float** gpyr_d, float** dogpyr_d, std::vector<Mat>& gpyr, std::vector<Mat>& dogpyr, float** itm_data, std::vector<int>& rows, std::vector<int>& cols, std::vector<cudaStream_t>& streams){
    CHECK(cudaDeviceSynchronize());
}

extern "C" void buildGaussianPyramidGPU(int octave_idx, Mat image, float** filter, float** gpyr, std::vector<Mat>& gpyr_mat , float** itm_data, std::vector<int>& rows, std::vector<int>& cols, std::vector<int>& filter_sizes, std::vector<cudaStream_t>& streams){
    for(int j = 0 ; j < nScales ; j++ ){
        if(octave_idx == 0 && j == 0){
        }else if(j == 0){
            halfImageGPU(gpyr[(octave_idx-1)*nScales + nOctaveLayers],gpyr[octave_idx*nScales],(float*)gpyr_mat[octave_idx*nScales+j].data,rows[octave_idx-1],cols[octave_idx-1],streams[j]);
        }else{
            gaussianBlur1DGPU(octave_idx,j,gpyr[octave_idx*nScales],gpyr[octave_idx*nScales+j],(float*)gpyr_mat[octave_idx*nScales+j].data,filter[j],itm_data[j],rows[octave_idx],cols[octave_idx],filter_sizes[j],streams[j]);
        }
    }
}
extern "C" void buildDoGPyramidGPU(int octave_idx, float** gpyr, float** dogpyr, std::vector<Mat>& dogpyr_mat, std::vector<int>& rows, std::vector<int>& cols, std::vector<cudaStream_t>& streams){
    for(int j = 0 ; j < nScales - 1 ; j++ ){
        differentiateGPU(gpyr[octave_idx*nScales+j],gpyr[octave_idx*nScales+j+1],dogpyr[octave_idx*(nScales - 1) + j],(float*)dogpyr_mat[octave_idx*(nScales-1)+j].data, rows[octave_idx], cols[octave_idx],streams[j]);
    }
}
