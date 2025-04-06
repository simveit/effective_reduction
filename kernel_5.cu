#include <cuda_runtime.h>

#include <cub/cub.cuh>
#include <iostream>

/*
Kernel 5
*/
void kernel_5_launch(const int *d_in, int *d_out, size_t N) {
  void* d_temp = nullptr;
  size_t temp_storage = 0;

  // First call to determine temporary storage size
  cub::DeviceReduce::Sum(d_temp, temp_storage, d_in, d_out, N);
  
  // Allocate temporary storage
  assert(temp_storage > 0);
  cudaMalloc(&d_temp, temp_storage);

  cub::DeviceReduce::Sum(d_temp, temp_storage, d_in, d_out, N);
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

  kernel_5_launch(d_in, d_out, N);

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
    kernel_5_launch(d_in, d_out, N);
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