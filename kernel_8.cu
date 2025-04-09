#include <cuda_runtime.h>

#include <cub/cub.cuh>
#include <iostream>

/*
Kernel 8
*/
template <unsigned int threadsPerBlock, unsigned int batchSize>
__global__ void kernel_8(const int4 *d_in, int *d_out, size_t N) {
  extern __shared__ int sums[threadsPerBlock];
  int sum = 0;
  const int tid = threadIdx.x;  
  const int global_tid = blockIdx.x * threadsPerBlock + tid;
  const int threads_in_grid = threadsPerBlock * gridDim.x;


  if (global_tid < N) {
#pragma unroll
    for (int i = 0; i < batchSize >> 2; i++) {
      const int4 val = d_in[global_tid * (batchSize >> 2) + i];
      if (global_tid * batchSize + i * 4 < N) {
        sum += val.x + val.y + val.z + val.w;
      }
    }
  }
  sums[tid] = sum;
  __syncthreads();

#pragma unroll
  for (int activeThreads = threadsPerBlock >> 1; activeThreads > 32;
       activeThreads >>= 1) {
    if (tid < activeThreads) {
      sums[tid] += sums[tid + activeThreads];
    }
    __syncthreads();
  }

  int sum_warp;

  if (tid < 32) {
    sum_warp = sums[tid] + sums[tid + 32];
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      sum_warp += __shfl_down_sync(0xffffffff, sum_warp, offset);
    }
  }

  if (tid == 0) {
    atomicAdd(d_out, sum_warp);
  }
}

template <int threadsPerBlock, int batchSize>
void kernel_8_launch(const int *d_in, int *d_out, size_t N) {
  const int numBlocks = (N + threadsPerBlock * batchSize - 1) /
                        (threadsPerBlock * batchSize);
  const int4 *d_in_cast = reinterpret_cast<const int4 *>(d_in);
  cudaMemset(d_out, 0, sizeof(int));
  kernel_8<threadsPerBlock, batchSize><<<numBlocks, threadsPerBlock>>>(d_in_cast,
                                                                       d_out, N);
}

__global__ void warmupKernel() { extern __shared__ int tmp[]; }

int main() {
  warmupKernel<<<1024, 1024, 1024 * sizeof(int)>>>();
  cudaDeviceSynchronize();

  const int N = 1 << 30;
  size_t size = N * sizeof(int);
  const int threadsPerBlock = 512;
  const int batchSize = 12;
  const int numBlocks = (N + threadsPerBlock * batchSize - 1) /
                        (threadsPerBlock * batchSize);

  int *h_in = new int[N];
  int *h_first = new int[numBlocks];
  int h_out = 0.0f;

  srand(42);
  for (int i = 0; i < N; ++i) {
    h_in[i] = rand() % 100;
  }

  int *d_in;
  int *d_first;
  int *d_out;
  cudaMalloc(&d_in, size);
  cudaMalloc(&d_first, numBlocks * sizeof(int));
  cudaMalloc(&d_out, sizeof(int));

  cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_first, h_first, numBlocks * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_out, &h_out, sizeof(int), cudaMemcpyHostToDevice);

  kernel_8_launch<threadsPerBlock, batchSize>(d_in, d_out, N);

  cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);

  int h_out_ref = 0;
  for (int i = 0; i < N; ++i) {
    h_out_ref += h_in[i];
  }
  std::cout << "h_out: " << h_out << ", h_out_ref: " << h_out_ref << std::endl;
  
  size_t num_runs = 1000;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  for (size_t i = 0; i < num_runs; ++i) {
    kernel_8_launch<threadsPerBlock, batchSize>(d_in, d_out, N);
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  milliseconds /= num_runs;
  std::cout << "Time taken: " << milliseconds << " ms" << std::endl;

  auto bandwidth = N * sizeof(int) / milliseconds / 1e6;
  const auto max_bandwidth = 3.3 * 1e3;  // 3.3 TB/s on H100

  std::cout << "Bandwidth: " << bandwidth << " GB/s" << std::endl;
  std::cout << "% of max bandwidth: " << bandwidth / max_bandwidth * 100
            << std::endl;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaFree(d_in);
  cudaFree(d_first);
  cudaFree(d_out);
  delete[] h_in;
}