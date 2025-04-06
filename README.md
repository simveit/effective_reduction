# effective_reduction
Improve reduction kernel step by step
Please see [my blogpost](https://veitner.bearblog.dev/making-vector-sum-really-fast/) for a detailed explanation.

`kernel_4` outperforms `kernel_5` which is the implementation of the NVIDIA library.

This is inspired by [fast.cu](https://github.com/pranjalssh/fast.cu/tree/main), a repo which aims to write highly performant CUDA kernels.
