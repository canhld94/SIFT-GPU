#include "sift.hpp"

/* Parameter to implement SIFT*/
static const int nOctaveLayers = 2; // number of layer per octaves
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
	std::vector<Mat> gpyr, dogpyr;
	double t, tf;
 	tf = getTickFrequency();
   	t = (double) getTickCount();
	Mat img = image.getMat();
	buildGaussianPyramid(img, gpyr, 5);
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

	t = (double) getTickCount();
	int dsize = SIFT_DESCR_WIDTH*SIFT_DESCR_WIDTH*SIFT_DESCR_HIST_BINS;
	descriptors.create((int)keypoints.size(), dsize, CV_32F);
	Mat _descriptors = descriptors.getMat();
	calDescriptor(gpyr, keypoints, _descriptors, 0);
	t = (double) getTickCount() - t;
	printf("descriptor extraction time: %g\n", t*1000./tf);

	return;
}

/* Gaussian Blur Function use 2D convolution */

static Mat getGaussianKernel(float sigma){
	data_t *coeff; // coeff array, row major
	int w = floor(3*sigma);
	int size = 2*w + 1;
	Mat gKernel(size, size, DATATYPE);
	coeff = (data_t *) gKernel.data;
	for (int i = -w; i <= w; ++i)
		for(int j = -w; j <= w; ++j){
			double dat = 1./(2*PI*sigma*sigma) * exp(-(i*i + j*j)*1./(2*sigma*sigma));
            dat = dat*8192;
			coeff[(i+w)*size + (j+w)] = (data_t) dat;
		}
	return gKernel;
}

static void getSubMatrix(int r, int c, Mat src, int ksize, data_t *data){
	data_t *imag = (data_t *) src.data;
	int w = floor(ksize/2);
	for (int i = -w; i <= w; ++i)
		for(int j = -w; j <= w; ++j){
			data_t ele;
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
	dst = Mat(rows, cols, DATATYPE);
	int ksize = gKernel.rows;
	// pad input image with 0
	data_t *image_data = new data_t[ksize*ksize];
	data_t *kernel_data = (data_t *) gKernel.data;
	data_t *output = (data_t *) dst.data;
	for (int i = 0; i < src.rows; ++i) {
		for (int j = 0; j < src.cols; ++j) {
			// fill image data
			getSubMatrix(i, j, src, ksize, image_data);
			// dot product
			float dotprod = 0;
			for(int k = 0; k < ksize*ksize; ++k){
				dotprod += image_data[k] * kernel_data[k];
			}
			output[i*dst.cols + j] = dotprod/8192;
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
	data_t *src_data = (data_t *) src.data;
	float *kernel_data = (float *) gKernel.data;
	data_t *itm_data = (data_t *) itm.data;
	data_t *output = (data_t *) dst.data;
	#pragma omp parallel num_threads(16) if(rows > 16) private(tid, nthreads, i, j)
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
	// Gaussian_Blur_1D(image, base, sigma);
	Gaussian_Blur(image, base, sigma);

	return base;
}

void buildGaussianPyramid(Mat& image,
						std::vector<Mat>& gpyr,
						int nOctaves){
	std::vector<float> sig(nScales);
	gpyr.resize(nOctaves*nScales);
	double k = pow(2.0, 1.0/nOctaveLayers);

	// create base image for first octave
	Mat base = createInitialImage(image, 0, sqrt(Sigma*Sigma+0.2*0.2));
    //  Mat base = createInitialImage(image, 0, sqrt(0.5*0.5));
	// pre-compute sigma for each scale
	sig[0] = Sigma;
	for(int i = 1; i < nScales; ++i){
		double sig_total = pow(k* 1.0, (double) i)*Sigma;
		sig[i] = (float) sqrt(sig_total*sig_total - Sigma*Sigma);
        // sig[i] = sig[i-1]*k;
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
				Gaussian_Blur(src, dst, sig[i]);
				// GaussianBlur(src, dst, Size(), sig[i], sig[i]);
			}
	}
	}
}

void buildDoGPyramid(std::vector<Mat>& gpyr,
					 std::vector<Mat>& dogpyr,
					 int nOctaves){
	dogpyr.resize(nOctaves*(nScales-1));
	int i, o;
    // #pragma omp parallel for private (o,i) collapse(2)
	for (o = 0; o < nOctaves; ++o) {
		for (i = 0; i < nScales - 1; ++i) {
			Mat& src0 = gpyr[o*nScales + i];
			Mat& src1 = gpyr[o*nScales + i + 1];
			Mat& dst = dogpyr[o*(nScales-1) + i];
			if(src0.size != src1.size){
				printf("Different input size at o = %d and i = %d, abort!\n", o ,i);
				exit(0);
			}
			dst = (src1 - src0);
		}
	}
}

/* Scale space extrema */

static bool adjustLocalExtrema( const std::vector<Mat>& dogpyr, KeyPoint& kpt, int octv,
                                int& layer, int& r, int& c, int nOctaveLayers,
                                float contrastThreshold, float edgeThreshold, float sigma )
{
    const float img_scale = 1./(255*SIFT_FIXPT_SCALE);
	// const float img_scale = 1.;
    const float deriv_scale = img_scale*0.5f;
    const float second_deriv_scale = img_scale;
    const float cross_deriv_scale = img_scale*0.25f;

    float xi=0, xr=0, xc=0, contr=0;
    int i = 0;

    for( ; i < SIFT_MAX_INTERP_STEPS; i++ )
    {
        int idx = octv*(nOctaveLayers+2) + layer;
        const Mat& img = dogpyr[idx];
        const Mat& prev = dogpyr[idx-1];
        const Mat& next = dogpyr[idx+1];

        Vec3f dD((img.at<data_t>(r, c+1) - img.at<data_t>(r, c-1))*deriv_scale,
                 (img.at<data_t>(r+1, c) - img.at<data_t>(r-1, c))*deriv_scale,
                 (next.at<data_t>(r, c) - prev.at<data_t>(r, c))*deriv_scale);

        float v2 = (float)img.at<data_t>(r, c)*2;
        float dxx = (img.at<data_t>(r, c+1) + img.at<data_t>(r, c-1) - v2)*second_deriv_scale;
        float dyy = (img.at<data_t>(r+1, c) + img.at<data_t>(r-1, c) - v2)*second_deriv_scale;
        float dss = (next.at<data_t>(r, c) + prev.at<data_t>(r, c) - v2)*second_deriv_scale;
        float dxy = (img.at<data_t>(r+1, c+1) - img.at<data_t>(r+1, c-1) -
                     img.at<data_t>(r-1, c+1) + img.at<data_t>(r-1, c-1))*cross_deriv_scale;
        float dxs = (next.at<data_t>(r, c+1) - next.at<data_t>(r, c-1) -
                     prev.at<data_t>(r, c+1) + prev.at<data_t>(r, c-1))*cross_deriv_scale;
        float dys = (next.at<data_t>(r+1, c) - next.at<data_t>(r-1, c) -
                     prev.at<data_t>(r+1, c) + prev.at<data_t>(r-1, c))*cross_deriv_scale;

        Matx33f H(dxx, dxy, dxs,
                  dxy, dyy, dys,
                  dxs, dys, dss);

        Vec3f X = H.solve(dD, DECOMP_LU);

        xi = -X[2];
        xr = -X[1];
        xc = -X[0];

        if( std::abs(xi) < 0.5f && std::abs(xr) < 0.5f && std::abs(xc) < 0.5f )
            break;

        if( std::abs(xi) > (float)(INT_MAX/3) ||
            std::abs(xr) > (float)(INT_MAX/3) ||
            std::abs(xc) > (float)(INT_MAX/3) )
            return false;

        c += cvRound(xc);
        r += cvRound(xr);
        layer += cvRound(xi);

        if( layer < 1 || layer > nOctaveLayers ||
            c < SIFT_IMG_BORDER || c >= img.cols - SIFT_IMG_BORDER  ||
            r < SIFT_IMG_BORDER || r >= img.rows - SIFT_IMG_BORDER )
            return false;
    }

    // ensure convergence of interpolation
    if( i >= SIFT_MAX_INTERP_STEPS )
        return false;

    {
        int idx = octv*(nOctaveLayers+2) + layer;
        const Mat& img = dogpyr[idx];
        const Mat& prev = dogpyr[idx-1];
        const Mat& next = dogpyr[idx+1];
        Matx31f dD((img.at<data_t>(r, c+1) - img.at<data_t>(r, c-1))*deriv_scale,
                   (img.at<data_t>(r+1, c) - img.at<data_t>(r-1, c))*deriv_scale,
                   (next.at<data_t>(r, c) - prev.at<data_t>(r, c))*deriv_scale);
        float t = dD.dot(Matx31f(xc, xr, xi));

        contr = img.at<data_t>(r, c)*img_scale + t * 0.5f;
        if( std::abs( contr ) * nOctaveLayers < contrastThreshold )
            return false;

        // principal curvatures are computed using the trace and det of Hessian
        float v2 = img.at<data_t>(r, c)*2.f;
        float dxx = (img.at<data_t>(r, c+1) + img.at<data_t>(r, c-1) - v2)*second_deriv_scale;
        float dyy = (img.at<data_t>(r+1, c) + img.at<data_t>(r-1, c) - v2)*second_deriv_scale;
        float dxy = (img.at<data_t>(r+1, c+1) - img.at<data_t>(r+1, c-1) -
                     img.at<data_t>(r-1, c+1) + img.at<data_t>(r-1, c-1)) * cross_deriv_scale;
        float tr = dxx + dyy;
        float det = dxx * dyy - dxy * dxy;

        if( det <= 0 || tr*tr*edgeThreshold >= (edgeThreshold + 1)*(edgeThreshold + 1)*det )
            return false;
    }

    kpt.pt.x = (c + xc) * (1 << octv);
    kpt.pt.y = (r + xr) * (1 << octv);
    kpt.octave = octv + (layer << 8) + (cvRound((xi + 0.5)*255) << 16);
    kpt.size = sigma*powf(2.f, (layer + xi) / nOctaveLayers)*(1 << octv)*2;
    kpt.response = std::abs(contr);

    return true;
}
static float calcOrientationHist( const Mat& img, Point pt, int radius,
                                  float sigma, float* hist, int n )
{
    int i, j, k, len = (radius*2+1)*(radius*2+1);

    float expf_scale = -1.f/(2.f * sigma * sigma);
    AutoBuffer<float> buf(len*4 + n+4);
    float *X = buf, *Y = X + len, *Mag = X, *Ori = Y + len, *W = Ori + len;
    float* temphist = W + len + 2;

    for( i = 0; i < n; i++ )
        temphist[i] = 0.f;

    for( i = -radius, k = 0; i <= radius; i++ )
    {
        int y = pt.y + i;
        if( y <= 0 || y >= img.rows - 1 )
            continue;
        for( j = -radius; j <= radius; j++ )
        {
            int x = pt.x + j;
            if( x <= 0 || x >= img.cols - 1 )
                continue;

            float dx = (float)(img.at<data_t>(y, x+1) - img.at<data_t>(y, x-1));
            float dy = (float)(img.at<data_t>(y-1, x) - img.at<data_t>(y+1, x));

            X[k] = dx; Y[k] = dy; W[k] = (i*i + j*j)*expf_scale;
            k++;
        }
    }

    len = k;

    // compute gradient values, orientations and the weights over the pixel neighborhood
    cv::hal::exp32f(W, W, len);
    cv::hal::fastAtan2(Y, X, Ori, len, true);
    cv::hal::magnitude32f(X, Y, Mag, len);

    k = 0;
    for( ; k < len; k++ )
    {
        int bin = cvRound((n/360.f)*Ori[k]);
        if( bin >= n )
            bin -= n;
        if( bin < 0 )
            bin += n;
        temphist[bin] += W[k]*Mag[k];
    }

    // smooth the histogram
    temphist[-1] = temphist[n-1];
    temphist[-2] = temphist[n-2];
    temphist[n] = temphist[0];
    temphist[n+1] = temphist[1];

    i = 0;
    for( ; i < n; i++ )
    {
        hist[i] = (temphist[i-2] + temphist[i+2])*(1.f/16.f) +
            (temphist[i-1] + temphist[i+1])*(4.f/16.f) +
            temphist[i]*(6.f/16.f);
    }

    float maxval = hist[0];
    for( i = 1; i < n; i++ )
        maxval = std::max(maxval, hist[i]);

    return maxval;
}



static void findScaleSpaceExtremaComputer(int o, // octave index
										 int i, // scales index
										 float threshold,
										 int idx,
										 int step,
										 int nOctaveLayers,
										 double contrastThreshold,
										 double edgeThreshold,
										 double sigma,
										 const std::vector<Mat>& gpyr,
										 const std::vector<Mat>& dogpyr,
										 TLSData<std::vector<KeyPoint> >&tls_kpts_struct)
{
	static const int n = SIFT_ORI_HIST_BINS;
	float hist[n];

	const Mat& img = dogpyr[idx];
	const Mat& prev = dogpyr[idx-1];
	const Mat& next = dogpyr[idx+1];
	int rows = img.rows;
	int cols = img.cols;

	std::vector<KeyPoint> *tls_kpts = tls_kpts_struct.get();

	KeyPoint kpt;
	for(int r = SIFT_IMG_BORDER; r < rows - SIFT_IMG_BORDER; ++r){
        const data_t* currptr = img.ptr<data_t>(r);
        const data_t* prevptr = prev.ptr<data_t>(r);
        const data_t* nextptr = next.ptr<data_t>(r);
		for(int c = SIFT_IMG_BORDER; c < cols - SIFT_IMG_BORDER; ++c){
			data_t val = currptr[c];
			if( std::abs(val) > threshold &&
			((val > 0 && val >= currptr[c-1] && val >= currptr[c+1] &&
			val >= currptr[c-step-1] && val >= currptr[c-step] && val >= currptr[c-step+1] &&
			val >= currptr[c+step-1] && val >= currptr[c+step] && val >= currptr[c+step+1] &&
			val >= nextptr[c] && val >= nextptr[c-1] && val >= nextptr[c+1] &&
			val >= nextptr[c-step-1] && val >= nextptr[c-step] && val >= nextptr[c-step+1] &&
			val >= nextptr[c+step-1] && val >= nextptr[c+step] && val >= nextptr[c+step+1] &&
			val >= prevptr[c] && val >= prevptr[c-1] && val >= prevptr[c+1] &&
			val >= prevptr[c-step-1] && val >= prevptr[c-step] && val >= prevptr[c-step+1] &&
			val >= prevptr[c+step-1] && val >= prevptr[c+step] && val >= prevptr[c+step+1]) ||
			(val < 0 && val <= currptr[c-1] && val <= currptr[c+1] &&
			val <= currptr[c-step-1] && val <= currptr[c-step] && val <= currptr[c-step+1] &&
			val <= currptr[c+step-1] && val <= currptr[c+step] && val <= currptr[c+step+1] &&
			val <= nextptr[c] && val <= nextptr[c-1] && val <= nextptr[c+1] &&
			val <= nextptr[c-step-1] && val <= nextptr[c-step] && val <= nextptr[c-step+1] &&
			val <= nextptr[c+step-1] && val <= nextptr[c+step] && val <= nextptr[c+step+1] &&
			val <= prevptr[c] && val <= prevptr[c-1] && val <= prevptr[c+1] &&
			val <= prevptr[c-step-1] && val <= prevptr[c-step] && val <= prevptr[c-step+1] &&
			val <= prevptr[c+step-1] && val <= prevptr[c+step] && val <= prevptr[c+step+1])))
			{
				int r1 = r, c1 = c, layer = i;
				if( !adjustLocalExtrema(dogpyr, kpt, o, layer, r1, c1,
										nOctaveLayers, (float)contrastThreshold,
										(float)edgeThreshold, (float)sigma) )
                        continue;
				float scl_octv = kpt.size*0.5f/(1 << o);
				float omax = calcOrientationHist(gpyr[o*(nOctaveLayers+3) + layer],
													Point(c1, r1),
													cvRound(SIFT_ORI_RADIUS * scl_octv),
													SIFT_ORI_SIG_FCTR * scl_octv,
													hist, n);
				float mag_thr = (float)(omax * SIFT_ORI_PEAK_RATIO);
				for( int j = 0; j < n; j++ )
				{
					int l = j > 0 ? j - 1 : n - 1;
					int r2 = j < n-1 ? j + 1 : 0;

					if( hist[j] > hist[l]  &&  hist[j] > hist[r2]  &&  hist[j] >= mag_thr )
					{
						float bin = j + 0.5f * (hist[l]-hist[r2]) / (hist[l] - 2*hist[j] + hist[r2]);
						bin = bin < 0 ? n + bin : bin >= n ? bin - n : bin;
						kpt.angle = 360.f - (float)((360.f/n) * bin);
						if(std::abs(kpt.angle - 360.f) < FLT_EPSILON)
							kpt.angle = 0.f;
						{
							tls_kpts->push_back(kpt);
						}
					}
				}
			}
		}
	}	
}

void findScaleSpaceExtrema(std::vector<Mat>& gpyr,
						   std::vector<Mat>& dogpyr,
						   std::vector<KeyPoint>& keypoints,
						   int nOctaves){
	const int threshold = cvFloor(0.5 * contrastThreshold / nOctaveLayers * 255);
	// const float threshold = 0.5 * contrastThreshold / nOctaveLayers;
    keypoints.clear();
    TLSData<std::vector<KeyPoint> > tls_kpts_struct;
    // #pragma omp parallel for collapse(2)
    for( int o = 0; o < nOctaves; o++ )
        for( int i = 1; i <= nOctaveLayers; i++ )
        {
            const int idx = o*(nOctaveLayers+2)+i;
            const Mat& img = dogpyr[idx];
            const int step = (int)img.step1();
            const int rows = img.rows, cols = img.cols;
			findScaleSpaceExtremaComputer(
                    o, i, 8, idx, step,
                    nOctaveLayers,
                    contrastThreshold,
                    edgeThreshold,
                    Sigma,
                    gpyr, dogpyr, tls_kpts_struct);
        }

    std::vector<std::vector<KeyPoint>*> kpt_vecs;
    tls_kpts_struct.gather(kpt_vecs);
    for (size_t i = 0; i < kpt_vecs.size(); ++i) {
        keypoints.insert(keypoints.end(), kpt_vecs[i]->begin(), kpt_vecs[i]->end());
    }
}

static void calcSIFTDescriptor( const Mat& img, Point2f ptf, float ori, float scl,
                               int d, int n, float* dst )
{
    Point pt(cvRound(ptf.x), cvRound(ptf.y));
    float cos_t = cosf(ori*(float)(CV_PI/180));
    float sin_t = sinf(ori*(float)(CV_PI/180));
    float bins_per_rad = n / 360.f;
    float exp_scale = -1.f/(d * d * 0.5f);
    float hist_width = SIFT_DESCR_SCL_FCTR * scl;
    int radius = cvRound(hist_width * 1.4142135623730951f * (d + 1) * 0.5f);
    // Clip the radius to the diagonal of the image to avoid autobuffer too large exception
    radius = std::min(radius, (int) sqrt(((double) img.cols)*img.cols + ((double) img.rows)*img.rows));
    cos_t /= hist_width;
    sin_t /= hist_width;

    int i, j, k, len = (radius*2+1)*(radius*2+1), histlen = (d+2)*(d+2)*(n+2);
    int rows = img.rows, cols = img.cols;

    AutoBuffer<float> buf(len*6 + histlen);
    float *X = buf, *Y = X + len, *Mag = Y, *Ori = Mag + len, *W = Ori + len;
    float *RBin = W + len, *CBin = RBin + len, *hist = CBin + len;

    for( i = 0; i < d+2; i++ )
    {
        for( j = 0; j < d+2; j++ )
            for( k = 0; k < n+2; k++ )
                hist[(i*(d+2) + j)*(n+2) + k] = 0.;
    }

    for( i = -radius, k = 0; i <= radius; i++ )
        for( j = -radius; j <= radius; j++ )
        {
            // Calculate sample's histogram array coords rotated relative to ori.
            // Subtract 0.5 so samples that fall e.g. in the center of row 1 (i.e.
            // r_rot = 1.5) have full weight placed in row 1 after interpolation.
            float c_rot = j * cos_t - i * sin_t;
            float r_rot = j * sin_t + i * cos_t;
            float rbin = r_rot + d/2 - 0.5f;
            float cbin = c_rot + d/2 - 0.5f;
            int r = pt.y + i, c = pt.x + j;

            if( rbin > -1 && rbin < d && cbin > -1 && cbin < d &&
                r > 0 && r < rows - 1 && c > 0 && c < cols - 1 )
            {
                float dx = (float)(img.at<data_t>(r, c+1) - img.at<data_t>(r, c-1));
                float dy = (float)(img.at<data_t>(r-1, c) - img.at<data_t>(r+1, c));
                X[k] = dx; Y[k] = dy; RBin[k] = rbin; CBin[k] = cbin;
                W[k] = (c_rot * c_rot + r_rot * r_rot)*exp_scale;
                k++;
            }
        }

    len = k;
    cv::hal::fastAtan2(Y, X, Ori, len, true);
    cv::hal::magnitude32f(X, Y, Mag, len);
    cv::hal::exp32f(W, W, len);

    k = 0;
    for( ; k < len; k++ )
    {
        float rbin = RBin[k], cbin = CBin[k];
        float obin = (Ori[k] - ori)*bins_per_rad;
        float mag = Mag[k]*W[k];

        int r0 = cvFloor( rbin );
        int c0 = cvFloor( cbin );
        int o0 = cvFloor( obin );
        rbin -= r0;
        cbin -= c0;
        obin -= o0;

        if( o0 < 0 )
            o0 += n;
        if( o0 >= n )
            o0 -= n;

        // histogram update using tri-linear interpolation
        float v_r1 = mag*rbin, v_r0 = mag - v_r1;
        float v_rc11 = v_r1*cbin, v_rc10 = v_r1 - v_rc11;
        float v_rc01 = v_r0*cbin, v_rc00 = v_r0 - v_rc01;
        float v_rco111 = v_rc11*obin, v_rco110 = v_rc11 - v_rco111;
        float v_rco101 = v_rc10*obin, v_rco100 = v_rc10 - v_rco101;
        float v_rco011 = v_rc01*obin, v_rco010 = v_rc01 - v_rco011;
        float v_rco001 = v_rc00*obin, v_rco000 = v_rc00 - v_rco001;

        int idx = ((r0+1)*(d+2) + c0+1)*(n+2) + o0;
        hist[idx] += v_rco000;
        hist[idx+1] += v_rco001;
        hist[idx+(n+2)] += v_rco010;
        hist[idx+(n+3)] += v_rco011;
        hist[idx+(d+2)*(n+2)] += v_rco100;
        hist[idx+(d+2)*(n+2)+1] += v_rco101;
        hist[idx+(d+3)*(n+2)] += v_rco110;
        hist[idx+(d+3)*(n+2)+1] += v_rco111;
    }

    // finalize histogram, since the orientation histograms are circular
    for( i = 0; i < d; i++ )
        for( j = 0; j < d; j++ )
        {
            int idx = ((i+1)*(d+2) + (j+1))*(n+2);
            hist[idx] += hist[idx+n];
            hist[idx+1] += hist[idx+n+1];
            for( k = 0; k < n; k++ )
                dst[(i*d + j)*n + k] = hist[idx+k];
        }
    // copy histogram to the descriptor,
    // apply hysteresis thresholding
    // and scale the result, so that it can be easily converted
    // to byte array
    float nrm2 = 0;
    len = d*d*n;
    k = 0;
    for( ; k < len; k++ )
        nrm2 += dst[k]*dst[k];

    float thr = std::sqrt(nrm2)*SIFT_DESCR_MAG_THR;

    i = 0, nrm2 = 0;
    for( ; i < len; i++ )
    {
        float val = std::min(dst[i], thr);
        dst[i] = val;
        nrm2 += val*val;
    }
    nrm2 = SIFT_INT_DESCR_FCTR/std::max(std::sqrt(nrm2), FLT_EPSILON);
    k = 0;

    for( ; k < len; k++ )
    {
        dst[k] = saturate_cast<uchar>(dst[k]*nrm2);
    }
    float nrm1 = 0;
    for( k = 0; k < len; k++ )
    {
        dst[k] *= nrm2;
        nrm1 += dst[k];
    }
    nrm1 = 1.f/std::max(nrm1, FLT_EPSILON);
    for( k = 0; k < len; k++ )
    {
        dst[k] = std::sqrt(dst[k] * nrm1);//saturate_cast<uchar>(std::sqrt(dst[k] * nrm1)*SIFT_INT_DESCR_FCTR);
    }
}

static inline void
unpackOctave(const KeyPoint& kpt, int& octave, int& layer, float& scale)
{
    octave = kpt.octave & 255;
    layer = (kpt.octave >> 8) & 255;
    octave = octave < 128 ? octave : (-128 | octave);
    scale = octave >= 0 ? 1.f/(1 << octave) : (float)(1 << -octave);
}

void calDescriptor( std::vector<Mat>& gpyr,
					std::vector<KeyPoint>& keypoints,
					Mat& descriptors,
					int firstOctave){
	static const int d = SIFT_DESCR_WIDTH, n = SIFT_DESCR_HIST_BINS;
    #pragma omp parallel for
	for(int i = 0; i < keypoints.size(); ++i){
		KeyPoint kpt = keypoints[i];
		int octave, layer;
		float scale;
		unpackOctave(kpt, octave, layer, scale);
		CV_Assert(octave >= firstOctave && layer <= nOctaveLayers + 2);
		float size = kpt.size*scale;
        Point2f ptf(kpt.pt.x*scale, kpt.pt.y*scale);
		const Mat& img = gpyr[(octave - firstOctave)*(nOctaveLayers + 3) + layer];
		float angle = 360.f - kpt.angle;
		if(std::abs(angle - 360.f) < FLT_EPSILON)
			angle = 0.f;
		calcSIFTDescriptor(img, ptf, angle, size*0.5f, d, n, descriptors.ptr<float>((int)i));		
	}
}