# effective_reduction
Improve reduction kernel step by step
Please see [my blogpost](https://veitner.bearblog.dev/making-vector-sum-really-fast/) for a detailed explanation.

This is inspired by [fast.cu](https://github.com/pranjalssh/fast.cu/tree/main), a repo which aims to write highly performant CUDA kernels.

`kernel_3`, `kernel_4`, `kernel_6`, `kernel_7`, `kernel_8` and `kernel_9` all outperform the NVIDIA CUB library for this problem.
The best performance is archieved for `kernel_9` which uses vectorization over batches as well as the `__reduce_add_sync` intrinsic.

## Performance Comparison

| Kernel | Bandwidth (GB/s) | % of Max Bandwidth | Implementation |
|--------|------------------|-------------------|----------------|
| kernel_0 | 639.31 | 19.37% | Custom |
| kernel_1 | 661.15 | 20.03% | Custom |
| kernel_2 | 859.24 | 26.04% | Custom |
| kernel_3 | 3228.89 | 97.85% | Custom |
| kernel_4 | 3231.46 | 97.92% | Custom |
| kernel_5 | 3190.48 | 96.68% | NVIDIA CUB |
| kernel_6 | 3230.55 | 97.90% | Custom |
| kernel_7 | 3230.98 | 97.91% | Custom |
| kernel_8 | 3232.48 | 97.95% | Custom |
| kernel_9 | 3233.34 | 97.98% | Custom |