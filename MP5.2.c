// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void scan(float *input, float *output, int len, int internalLayer) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  int x;
  int t = threadIdx.x;
  int b = blockIdx.x;
  int loadStride;
  if (!internalLayer) {
    x = 2*b*blockDim.x + t;
    loadStride = blockDim.x;
  } else {
    x = (t+1)*(blockDim.x*2)-1;
    loadStride = BLOCK_SIZE*blockDim.x*2;
  }
  
  int storeX = 2*b*blockDim.x + t;
  
  __shared__ float T[2*BLOCK_SIZE];

  T[t] = x < len ? input[x] : 0;
  T[t+blockDim.x] = x + loadStride < len ? input[x+loadStride] : 0;

  int stride = 1;
  while(stride < 2*BLOCK_SIZE)
  {
    __syncthreads();
    int index = (threadIdx.x+1)*stride*2 - 1;
    if(index < 2*BLOCK_SIZE && (index-stride) >=0)
      T[index] += T[index-stride];
    stride = stride*2;
  }
  
  stride = BLOCK_SIZE/2;
  while(stride > 0)
  {
    __syncthreads();
    int index = (threadIdx.x+1)*stride*2 - 1;
    if((index+stride) < 2*BLOCK_SIZE) {
      T[index+stride] += T[index];
    }				
    stride = stride / 2;
  }	
  
  __syncthreads();
  if(storeX < len)
    output[storeX]=T[t];
  if(storeX + blockDim.x < len)
    output[storeX+blockDim.x]=T[t+blockDim.x];

}

__global__ void add(float* input, float *output, float*sum, int len) {
  int x = threadIdx.x + (blockIdx.x * blockDim.x * 2);
  __shared__ float increment;
  if (threadIdx.x == 0)
    increment = blockIdx.x == 0 ? 0 : sum[blockIdx.x - 1];
  __syncthreads();
  if (x < len)
    output[x] = input[x] + increment;
  if (x + blockDim.x < len)
    output[x + blockDim.x] = input[x + blockDim.x] + increment;
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceScanBuffer;
  float *deviceScanSums;
  float *deviceOutput;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceScanBuffer, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceScanSums,   2 * BLOCK_SIZE * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce

  dim3 dimGrid(ceil(numElements/float(BLOCK_SIZE*2)), 1, 1);
  dim3 dimBlock(BLOCK_SIZE, 1, 1);
  scan<<<dimGrid, dimBlock>>>(deviceInput, deviceScanBuffer, numElements, 0);
  cudaDeviceSynchronize();
  
  dim3 singleGrid(1, 1, 1);
  scan<<<singleGrid, dimBlock>>>(deviceScanBuffer, deviceScanSums, numElements, 1);
  cudaDeviceSynchronize();

  add<<<dimGrid, dimBlock>>>(deviceScanBuffer, deviceOutput, deviceScanSums, numElements);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(deviceScanBuffer);
  cudaFree(deviceScanSums);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
