# effective_reduction
Improve reduction kernel step by step
Please see [my blogpost](https://veitner.bearblog.dev/making-vector-sum-really-fast/) for a detailed explanation.

This is inspired by [fast.cu](https://github.com/pranjalssh/fast.cu/tree/main), a repo which aims to write highly performant CUDA kernels.

`kernel_3`, `kernel_4`, `kernel_6`, `kernel_7`, `kernel_8` and `kernel_9` all outperform the NVIDIA CUB library for this problem.
The best performance is archieved for `kernel_9` which uses vectorization over batches as well as the `__reduce_add_sync` intrinsic.

## Performance Comparison

| Kernel | Bandwidth (GB/s) | % of Max Bandwidth | Implementation |
|--------|------------------|-------------------|----------------|
| kernel_0 | 639.37 | 19.37% | Custom |
| kernel_1 | 661.16 | 20.04% | Custom |
| kernel_2 | 882.823 | 26.75% | Custom |
| kernel_3 | 3226.86 | 97.78% | Custom |
| kernel_4 | 3229.98 | 97.88% | Custom |
| kernel_5 | 3190.53 | 96.68% | NVIDIA CUB |
| kernel_6 | 3229.01 | 97.85% | Custom |
| kernel_7 | 3229.62 | 97.87% | Custom |
| kernel_8 | 3231.03 | 97.91% | Custom |
| kernel_9 | 3231.54 | 97.93% | Custom |
