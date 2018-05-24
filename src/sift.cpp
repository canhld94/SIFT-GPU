#include "sift.hpp"

/* Parameter to implement SIFT*/
const int nOctaveLayers = 2; // number of layer per octaves
const int nScales = 5; // nOctaveLayers + 3
const double Sigma = 1.6; // initial sigma
const double PI = 3.14159265359;



void SITF_BuildIn_OpenCV(InputArray image,
						 std::vector<KeyPoint>& keypoints,
						 OutputArray descriptors){
	Ptr<SIFT> detector = SIFT::create();
	Mat tmp;
	detector->detectAndCompute(image, tmp, keypoints, descriptors, 0);
	detector->clear();
	return;
}


void SIFT_NCL(InputArray image,
		  std::vector<KeyPoint> & keypoints,
		  OutputArray descriptors){

}

/* Gaussian Blur Function use 2D convolution */

static Mat getGaussianKernel(float sigma){
	float *coeff; // coeff array, row major
	int w = floor(3*sigma);
	int size = 2*w + 1;
	Mat gKernel(size, size, CV_32F);
	coeff = (float *) gKernel.data;
	for (int i = -w; i <= w; ++i)
		for(int j = -w; j <= w; ++j){
			double dat = 1./(2*PI*sigma*sigma) * exp(-(i*i + j*j)*1./(2*sigma*sigma));
			coeff[(i+w)*size + (j+w)] = (float) dat;
		}
	return gKernel;
}

static void getSubMatrix(int r, int c, Mat src, int ksize, float *data){
	float *imag = (float *) src.data;
	int w = floor(ksize/2);
	for (int i = -w; i <= w; ++i)
		for(int j = -w; j <= w; ++j){
			float ele;
			if(r+i < 0 || c+j < 0 || r+i >= src.rows-1 || c+j >= src.cols-1) ele = 0;
			else ele = imag[(r+i)*src.cols + c+j];
			data[(i+w)*ksize + (j+w)] = ele;
			}
}


void Gaussian_Blur(Mat& src, 
				   Mat& dst, 
				   double sigma){
	double t, tf = getTickFrequency();
	t = (double) getTickCount();
	Mat gKernel = getGaussianKernel(sigma);
	int rows = src.rows;
	int cols = src.cols;
	dst = Mat(rows, cols, CV_32F);
	int ksize = gKernel.rows;
	// pad input image with 0
	float *image_data = new float[ksize*ksize];
	float *kernel_data = (float *) gKernel.data;
	float *output = (float *) dst.data;
	for (int i = 0; i < src.rows; ++i) {
		for (int j = 0; j < src.cols; ++j) {
			// fill image data
			getSubMatrix(i, j, src, ksize, image_data);
			// dot product
			float dotprod = 0;
			for(int k = 0; k < ksize*ksize; ++k){
				dotprod += image_data[k] * kernel_data[k];
			}
			output[i*dst.cols + j] = dotprod;
			if(dotprod > 255 ) printf("error in dot prdo %d %d\n", i, j);
		}

	}
	t = (double) getTickCount() - t;
	printf("gaussian Blur time: %g\n", t*1000./tf);
	delete [] image_data;
}

 /* Gaussian Blur Function use 1-D convolution */

static Mat getGaussianKernel1D(double sigma) {
	float *coeff; // coeff array, row major
	int w = floor(3*sigma);
	int size = 2*w + 1;
	Mat gKernel(1, size, CV_32F);
	coeff = (float *) gKernel.data;
	for (int i = -w; i <= w; ++i){
			double dat = 1./sqrt(2*PI*sigma*sigma) * exp(-((double)i*i)*1./(2*sigma*sigma));
			coeff[i+w] = (float) dat;
		}
	return gKernel;
}

void Gaussian_Blur_1D(Mat& src, 
					  Mat& dst, 
					  double sigma){
	// double t, tf = getTickFrequency();
	// t = (double) getTickCount();
	Mat gKernel = getGaussianKernel1D(sigma);
	int rows = src.rows;
	int cols = src.cols;
	dst = Mat(rows, cols, CV_32F);
	Mat itm = Mat(rows, cols, CV_32F); // intermediate image
	int ksize = gKernel.cols;
	int tid, nthreads, i, j;
	float *src_data = (float *) src.data;
	float *kernel_data = (float *) gKernel.data;
	float *itm_data = (float *) itm.data;
	float *output = (float *) dst.data;
	#pragma omp parallel num_threads(64) if(rows > 64) private(tid, nthreads, i, j)
	{
	tid = omp_get_thread_num();
	nthreads = omp_get_num_threads();
	int work_load = rows/nthreads;
	int top_bound = (tid == nthreads-1)?rows:(tid+1)*work_load;
	// vertical convolution
	for (i = tid*work_load; i < top_bound; ++i) {
		for (j = 0; j < cols; ++j) {
			float dotprod = 0;
			for(int k = -ksize/2; k < ksize/2; ++k){
				dotprod += (i+k < 0 || i+k >= rows-1)?0:src_data[(i+k)*cols + j] * kernel_data[k+ksize/2];
			}
			itm_data[i*cols + j] = dotprod;
		}
	}
	// #pragma omp barrier
	// horizontal convolution
	for (i = tid*work_load; i < top_bound; ++i) {
		for (j = 0; j < cols; ++j) {
			float dotprod = 0;
			for(int k = -ksize/2; k < ksize/2; ++k){
				dotprod += (j+k < 0 || j+k >= cols-1)?0:itm_data[i*cols + j+k] * kernel_data[k+ksize/2];
			}
			output[i*cols + j] = dotprod;
		}
	}
	}
	itm.release();
	// t = (double) getTickCount() - t;
	// printf("gaussian Blur time: %g\n", t*1000./tf);
}

static Mat createInitialImage(Mat& image,
							  bool doubleSize,
					   		  double sigma){
	Mat base;
	Gaussian_Blur_1D(image, base, sigma);
	return base;
}

void buildGaussianPyramid(Mat& image,
						std::vector<Mat>& gpyr,
						int nOctaves){
	std::vector<float> sig(nScales);
	gpyr.resize(nOctaves*nScales);
	double k = pow(2.0, 1.0/nOctaveLayers);

	// create base image for first octave
	Mat base = createInitialImage(image, 0, sqrt(Sigma*Sigma + 0.5*0.5));
	// pre-compute sigma for each scale
	sig[0] = Sigma;
	for(int i = 1; i < nScales; ++i){
		double sig_total = pow(k* 1.0, (double) i)*Sigma;
		sig[i] = (float) sqrt(sig_total*sig_total - Sigma*Sigma);
	}
	for (int o = 0; o < nOctaves; ++o) {
		for (int i = 0; i < nScales; ++i) {
			Mat& dst = gpyr[o*nOctaves + i];
			if(o == 0 && i == 0) {// first base
				dst = base;
			}
			else if(i == 0) { // first scale of next octave is constructed from last
				Mat& src = gpyr[(o-1)*nScales + nOctaveLayers]; // last gaussian image
				resize(src, dst,Size(src.cols/2, src.rows/2), 0, 0, INTER_NEAREST);
			}
			else{
				Mat& src = gpyr[o*nScales]; // Base of current octave
				Gaussian_Blur_1D(src, dst, sig[i]);
				// GaussianBlur(src, dst, Size(), sig[i], sig[i]);
			}
	}
	}
}

void buildDoGPyramid(std::vector<Mat>& gpyr,
					 std::vector<Mat>& dogpyr,
					 int nOctaves){
	dogpyr.resize(nOctaves*(nScales-1));
	for (int o = 0; o < nOctaves; ++o) {
		for (int i = 0; i < nScales - 1; ++i) {
			Mat& src0 = gpyr[o*nScales + i];
			Mat& src1 = gpyr[o*nScales + i + 1];
			Mat& dst = dogpyr[o*(nScales-1) + i];
			if(src0.size != src1.size){
				printf("Different input size at o = %d and i = %d, abort!\n", o ,i);
				return;
			}
			dst = src1 - src0;
		}
	}
}

/* Scale space extrema */

static void findScaleSpaceExtremaCompute(int o, // octave index
										 int i, // scales index
										 int threshold,
										 int idx




){

}

void findScaleSpaceExtrema(std::vector<Mat>& gpyr,
						   std::vector<Mat>& dogpyr,
						   std::vector<KeyPoint>& keypoints){
	
}

