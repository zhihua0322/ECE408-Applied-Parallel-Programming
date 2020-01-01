// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 32

//@@ insert code here
__global__ void Cast(float *input, unsigned char *output, int width, int height) {

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    int idx = blockIdx.z * (width * height) + y * (width) + x;
    output[idx] = (unsigned char) ((HISTOGRAM_LENGTH - 1) * input[idx]);
  }
}

__global__ void Convert(unsigned char *ucharImage, unsigned char *grayImage, int width, int height) {
  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  int jj = blockIdx.y * blockDim.y + threadIdx.y;
  if (ii < width && jj < height) {
    int idx = jj *width + ii;
    unsigned char r = ucharImage[3*idx];
    unsigned char g = ucharImage[3*idx + 1];
    unsigned char b = ucharImage[3*idx + 2];
    grayImage[idx] = (unsigned char) (0.21*r + 0.71*g + 0.07*b);
  }
}

__global__ void Histogram(unsigned char *input, unsigned int *output, int width, int height) {

  __shared__ unsigned int histogram[HISTOGRAM_LENGTH];

  int tIdx = threadIdx.x + threadIdx.y * blockDim.x;
  if (tIdx < HISTOGRAM_LENGTH) {
    histogram[tIdx] = 0;
  }

  __syncthreads();
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height) {
    int idx = y * (width) + x;
    unsigned char val = input[idx];
    atomicAdd(&(histogram[val]), 1);
  }

  __syncthreads();
  if (tIdx < HISTOGRAM_LENGTH) {
    atomicAdd(&(output[tIdx]), histogram[tIdx]);
  }
}

__global__ void Scan(unsigned int *input, float *output, int width, int height) {
  int x = threadIdx.x;
    
  __shared__ unsigned int T[HISTOGRAM_LENGTH];

  T[x] = input[x];

  int stride = 1;
  while(stride <= HISTOGRAM_LENGTH /2)
  {
    __syncthreads();
    int index = (threadIdx.x+1)*stride*2 - 1;
    if(index < HISTOGRAM_LENGTH)
      T[index] += T[index-stride];
    stride = stride*2;
  }
  
  stride = HISTOGRAM_LENGTH/4;
  while(stride > 0)
  {
    __syncthreads();
    int index = (threadIdx.x+1)*stride*2 - 1;
    if((index+stride) < HISTOGRAM_LENGTH) {
      T[index+stride] += T[index];
    }
    stride = stride / 2;
  }	
  
  __syncthreads();
  output[x] = T[x] / ((float) (width * height));

}

__global__ void Equalize(unsigned char *inout, float *cdf, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    int idx = blockIdx.z * (width * height) + y * (width) + x;
    unsigned char val = inout[idx];

    float equalized = 255 * (cdf[val] - cdf[0]) / (1.0 - cdf[0]);
    float clamped   = min(max(equalized, 0.0), 255.0);

    inout[idx] = (unsigned char) (clamped);
  }
}

__global__ void CastBack(unsigned char *input, float *output,
                              int width, int height) {

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    int idx = blockIdx.z * (width * height) + y * (width) + x;
    output[idx] = (float) (input[idx] / 255.0);
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;
  
  float *floatImage;
  unsigned char *ucharImage;
  unsigned char *grayImage;
  unsigned int *histogram;
  float *cdf;
  //@@ Insert more code here

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);
  
  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  hostInputImageData  = wbImage_getData(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");
 
  wbTime_start(GPU, "Allocating GPU memory.");
  int floatImageSize = imageWidth * imageHeight * imageChannels * sizeof(float);
  int ucharImageSize = imageWidth * imageHeight * imageChannels * sizeof(unsigned char);
  int grayImageSize = imageWidth * imageHeight * sizeof(unsigned char);
  int histogramSize = HISTOGRAM_LENGTH * sizeof(unsigned int);
  int cdfSize = HISTOGRAM_LENGTH * sizeof(float);
  cudaMalloc((void **)&floatImage, floatImageSize);
  cudaMalloc((void **)&ucharImage, ucharImageSize);
  cudaMalloc((void **)&grayImage, grayImageSize);
  cudaMalloc((void **)&histogram, histogramSize);
  cudaMemset((void *) histogram, 0, histogramSize);
  cudaMalloc((void **)&cdf, cdfSize);
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  cudaMemcpy(floatImage, hostInputImageData, floatImageSize, cudaMemcpyHostToDevice);
  wbTime_stop(GPU, "Copying input memory to the GPU.");
  //@@ insert code here
  

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  dim3 DimGrid1(ceil(imageWidth/double(BLOCK_SIZE)), ceil(imageHeight/double(BLOCK_SIZE)), imageChannels);
  dim3 DimBlock1(BLOCK_SIZE, BLOCK_SIZE, 1);
  Cast<<<DimGrid1, DimBlock1>>>(floatImage, ucharImage, imageWidth, imageHeight);
  cudaDeviceSynchronize();
  
  dim3 DimGrid2(ceil(imageWidth/double(BLOCK_SIZE)), ceil(imageHeight/double(BLOCK_SIZE)), 1);
  dim3 DimBlock2(BLOCK_SIZE, BLOCK_SIZE, 1);
  Convert<<<DimGrid2, DimBlock2>>>(ucharImage, grayImage, imageWidth, imageHeight);
  cudaDeviceSynchronize();
 
  dim3 DimGrid3(ceil(imageWidth/double(BLOCK_SIZE)), ceil(imageHeight/double(BLOCK_SIZE)), 1);
  dim3 DimBlock3(BLOCK_SIZE, BLOCK_SIZE, 1);
  Histogram<<<DimGrid3, DimBlock3>>>(grayImage, histogram, imageWidth, imageHeight);
  cudaDeviceSynchronize();
  
  dim3 DimGrid4(1, 1, 1);
  dim3 DimBlock4(HISTOGRAM_LENGTH, 1, 1);
  Scan<<<DimGrid4, DimBlock4>>>(histogram, cdf, imageWidth, imageHeight);
  cudaDeviceSynchronize();
 
  dim3 DimGrid5(ceil(imageWidth/double(BLOCK_SIZE)), ceil(imageHeight/double(BLOCK_SIZE)), imageChannels);
  dim3 DimBlock5(BLOCK_SIZE, BLOCK_SIZE, 1);
  Equalize<<<DimGrid5, DimBlock5>>>(ucharImage, cdf, imageWidth, imageHeight);
  cudaDeviceSynchronize();
  
  
  dim3 DimGrid6(ceil(imageWidth/double(BLOCK_SIZE)), ceil(imageHeight/double(BLOCK_SIZE)), imageChannels);
  dim3 DimBlock6(BLOCK_SIZE, BLOCK_SIZE, 1);
  CastBack<<<DimGrid6, DimBlock6>>>(ucharImage, floatImage, imageWidth, imageHeight);
  cudaDeviceSynchronize();
  
  wbTime_stop(Compute, "Performing CUDA computation");
  cudaMemcpy(hostOutputImageData, floatImage, floatImageSize, cudaMemcpyDeviceToHost);

  wbSolution(args, outputImage);
  cudaFree(floatImage);

  //@@ insert code here

  return 0;
}
