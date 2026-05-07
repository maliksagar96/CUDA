#include <iostream>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

using namespace std;

#define CUDA_CHECK(call)                                                   \
do {                                                                       \
  cudaError_t err = call;                                                  \
  if(err != cudaSuccess) {                                                 \
    cerr << "CUDA Error: " << cudaGetStringError(err)                      \
         << " at line " << __LINE__ << endl;                               \
    exit(EXIT_FAILURE);                                                    \
  }                                                                        \
} while(0)

#define BLOCK_SIZE (16u)
#define FILTER_SIZE (5u)
#define TILE_SIZE (12u)

__global__ void process_image(unsigned char *out, unsigned char *in, size_t pitch, unsigned int Width, unsigned int Height) {

	//Coordinates of output pixels.
	int x_o = (TILE_SIZE * blockIdx.x) + threadIdx.x;
	int y_o = (TILE_SIZE * blockIdx.y) + threadIdx.y;

	//Input coordinate of the shared memory 
	int x_i = x_o - FILTER_SIZE/2;
	int y_i = y_o - FILTER_SIZE/2;

	int sum = 0;

	//define shared memory
	__shared__ unsigned char sBuffer[BLOCK_SIZE][BLOCK_SIZE];

	//copy inside shared memory 
	if(x_i >= 0 && x_i < Width && y_i >= 0 && y_i < Height) {
		sBuffer[threadIdx.y][threadIdx.x] = in[y_i * pitch + x_i];
	}

	else {
		sBuffer[threadIdx.y][threadIdx.x] = 0;		
	}

	__syncthreads();

	if(threadIdx.x < TILE_SIZE && threadIdx.y < TILE_SIZE) {
		for(int r = 0;r < FILTER_SIZE;r++) {
			for(int c = 0;c < FILTER_SIZE;c++){
				sum += sBuffer[threadIdx.y + r][threadIdx.x + c];
			}
		}

		sum /= (FILTER_SIZE * FILTER_SIZE);
		
		
		if(x_o < Width && y_o < Height) {
			out[y_o * Width + x_o] = sum;
		}	
	}

}

int main() {

	cout << "Loading the image ...\n";
  cv::Mat img = cv::imread("../output.png");
  if(img.empty()) {
    std::cout << "Image not loaded\n";
    return 1;
  }

	cout << "Image loaded.\n";
	int Width = img.cols;
	int Height = img.rows;

  std::cout << "Width  : " << Width << '\n';
  std::cout << "Height : " << Height << '\n';

	int size = Width * Height;
	int byteSize = size * sizeof(unsigned char);

	//reading the color pixel value
	vector<unsigned char> h_r(size), h_r_n(size);
	vector<unsigned char> h_g(size), h_g_n(size);
	vector<unsigned char> h_b(size), h_b_n(size);

	for(int y = 0; y < Height; y++) {
		for(int x = 0; x < Width; x++) {
			cv::Vec3b pixel = img.at<cv::Vec3b>(y, x);
			int idx = y * Width + x;
			h_b[idx] = pixel[0];
			h_g[idx] = pixel[1];
			h_r[idx] = pixel[2];
		}
	}

	unsigned char *d_r_n, *d_g_n, *d_b_n;

	cudaMalloc(&d_r_n, byteSize);
	cudaMalloc(&d_g_n, byteSize);
	cudaMalloc(&d_b_n, byteSize);

	unsigned char *d_r, *d_g, *d_b;

	size_t pitch_r, pitch_g, pitch_b;

	cudaMallocPitch(&d_r, &pitch_r, Width, Height);
	cudaMallocPitch(&d_g, &pitch_g, Width, Height);
	cudaMallocPitch(&d_b, &pitch_b, Width, Height);

	cudaMemcpy2D(d_r, pitch_r, h_r.data(), Width, Width, Height, cudaMemcpyHostToDevice);
	cudaMemcpy2D(d_g, pitch_g, h_g.data(), Width, Width, Height, cudaMemcpyHostToDevice);
	cudaMemcpy2D(d_b, pitch_b, h_b.data(), Width, Width, Height, cudaMemcpyHostToDevice);

	dim3 grid((Width + TILE_SIZE - 1)/TILE_SIZE, (Height + TILE_SIZE - 1)/TILE_SIZE);
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);

	process_image<<<grid, block>>>(d_r_n, d_r, pitch_r, Width, Height);
	process_image<<<grid, block>>>(d_g_n, d_g, pitch_g, Width, Height);
	process_image<<<grid, block>>>(d_b_n, d_b, pitch_b, Width, Height);
	cudaDeviceSynchronize();

	cudaMemcpy(h_r_n.data(), d_r_n, byteSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_g_n.data(), d_g_n, byteSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_b_n.data(), d_b_n, byteSize, cudaMemcpyDeviceToHost);

	cv::Mat output(Height, Width, CV_8UC3);

	for(int y = 0; y < Height; y++) {
		for(int x = 0; x < Width; x++) {
			
			int idx = y * Width + x;
			output.at<cv::Vec3b>(y, x)[0] = h_b_n[idx];
			output.at<cv::Vec3b>(y, x)[1] = h_g_n[idx];
			output.at<cv::Vec3b>(y, x)[2] = h_r_n[idx];
		}
	}

	cv::imwrite("../output2.png", output);
	cudaFree(d_r); cudaFree(d_g); cudaFree(d_b);
	cudaFree(d_r_n); cudaFree(d_g_n); cudaFree(d_b_n);

  return 0;
}