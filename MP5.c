#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define TILE_WIDTH 4
#define MASK_WIDTH 3
#define MASK_RADIUS 1

#define inBounds(x, y, z) \
  ((0 <= (x) && (x) < x_size) && \
   (0 <= (y) && (y) < y_size) && \
   (0 <= (z) && (z) < z_size))
//@@ Define constant memory for device kernel here
__constant__ float deviceKernel[MASK_WIDTH][MASK_WIDTH][MASK_WIDTH];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  int bx = blockIdx.x * TILE_WIDTH; int tx = threadIdx.x;
  int by = blockIdx.y * TILE_WIDTH; int ty = threadIdx.y;
  int bz = blockIdx.z * TILE_WIDTH; int tz = threadIdx.z;
  int x_o = bx + tx;
  int y_o = by + ty;
  int z_o = bz + tz;
  int x_i = x_o - MASK_RADIUS;
  int y_i = y_o - MASK_RADIUS;
  int z_i = z_o - MASK_RADIUS;
  __shared__ float N_ds[TILE_WIDTH + MASK_WIDTH - 1][TILE_WIDTH + MASK_WIDTH - 1][TILE_WIDTH + MASK_WIDTH - 1];
  if(inBounds(x_i, y_i, z_i)) {
    N_ds[tz][ty][tx] = input[z_i * x_size * y_size + y_i * x_size + x_i];
  } else {
    N_ds[tz][ty][tx] = 0;
  }
  __syncthreads();
  float sum = 0.0f;
  if (tx < TILE_WIDTH && ty < TILE_WIDTH && tz < TILE_WIDTH) {
    for(int i = 0; i < MASK_WIDTH; i++) {
      for(int j = 0; j < MASK_WIDTH; j++) {
        for(int n = 0; n < MASK_WIDTH; n++) {
          sum += deviceKernel[i][j][n] * N_ds[i + tz][j + ty][n + tx];
        }
      }
    }
    if (x_o < x_size && y_o < y_size && z_o < z_size) {
      output[z_o * x_size * y_size + y_o * x_size + x_o] = sum;
    }
  }
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel = (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  int deviceSize = (inputLength - 3) * sizeof(float);  
  cudaMalloc((void **) &deviceInput, deviceSize);
  cudaMalloc((void **) &deviceOutput, deviceSize);
  wbTime_stop(GPU, "Doing GPU memory allocation");
  
  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  cudaMemcpy(deviceInput, &hostInput[3], deviceSize, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(deviceKernel, hostKernel, kernelLength * sizeof(float));
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  dim3 dimGrid(ceil(x_size/double(TILE_WIDTH)), ceil(y_size/double(TILE_WIDTH)), ceil(z_size/double(TILE_WIDTH)));
  dim3 dimBlock(TILE_WIDTH + MASK_WIDTH - 1, TILE_WIDTH + MASK_WIDTH - 1, TILE_WIDTH + MASK_WIDTH - 1);
  //@@ Launch the GPU kernel here
  conv3d<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy(&hostOutput[3], deviceOutput, deviceSize, cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}