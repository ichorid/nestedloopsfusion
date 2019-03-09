#include <stdio.h>
#include <stdlib.h>
// CUDA runtime
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>
#define BLOCKSIZE 5
typedef unsigned short uint16;

__device__ unsigned int rolMaskA(unsigned int value, unsigned int amount) {
    return (value << amount) | (value >> ((32 - amount) & 31));
}

__device__ unsigned int lanemask_eq() {
    unsigned int mask;
    asm("mov.u32 %0, %%lanemask_eq;" : "=r"(mask));
    return mask;
}

__device__ unsigned int lfsr(uint16 start_state, int cycles = 1) {
    uint16 lfsr = start_state;
    uint16 bit; /* Must be 16bit to allow bit<<15 later in the code */
    for (int i = 0; i < cycles; i++) {
        /* taps: 16 14 13 11; feedback polynomial: x^16
         * + x^14 + x^13 + x^11 + 1 */
        bit = ((lfsr >> 0) ^ (lfsr >> 2) ^ (lfsr >> 3) ^ (lfsr >> 5)) & 1;
        lfsr = (lfsr >> 1) | (bit << 15);
    }
    return lfsr;
}
__global__ void kernel_gpu(
        int *out,
        unsigned int seed,
        int external_loop_iterations,
        unsigned int exit_mask_width,
        const int int_lfsr_cycles,
        const int repeat_times) {
    const unsigned tid = (blockIdx.x << BLOCKSIZE) + threadIdx.x;
    const unsigned int lmask = lanemask_eq();
    int total_cnt = 0;
    const unsigned int tmask = 0xffffffff >> (32 - exit_mask_width);
    for (int k=0; k < repeat_times; k++){
        uint16 rnd_common = lfsr(seed + 3457, 100);
        uint16 o_cnt = 0;
        do {
            int i_cnt = 0;
            o_cnt++;
            rnd_common = lfsr(rnd_common);
            const unsigned int exit_mask = rolMaskA(tmask, rnd_common & 31);

            uint16 irnd = lfsr(rnd_common ^ tid + 321, 3);
            do {
                i_cnt++;
                total_cnt += (irnd & 1);
                irnd = lfsr(irnd, int_lfsr_cycles);
            } while (!((lmask & exit_mask)) && (i_cnt < 10000));
        } while (o_cnt < external_loop_iterations);
    }
    out[tid] = total_cnt;
}

int main(int argc, char *argv[]) {

    int repeat_times = atoi(argv[1]);
    uint16 seed = atoi(argv[2]);
    int external_loop_iterations = atoi(argv[3]);
    unsigned int zt = strtoul(argv[4], NULL, 0);
    int int_lfsr_cycles = atoi(argv[5]);

    int *D_out;

    int num_blocks = 1 << 0;
    int num_threads = num_blocks * (1 << BLOCKSIZE);
    cudaEvent_t start, stop;
    checkCudaErrors(cudaMalloc((void **)&D_out, num_threads * sizeof(int)));
    cudaEventCreate(&start);
    cudaEventRecord(start, 0);
    kernel_gpu<<<num_blocks, (1 << BLOCKSIZE)>>>(D_out, seed, external_loop_iterations, zt,
            int_lfsr_cycles, repeat_times);
    cudaDeviceSynchronize();
    cudaEventCreate(&stop);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    double tm = elapsedTime / 1000;

    int *H_out = (int *)malloc(num_threads * sizeof(int));
    checkCudaErrors(cudaMemcpy((void *)H_out, D_out, num_threads * sizeof(int),
                cudaMemcpyDeviceToHost));
    long int sum = 0;
    for (int i = 0; i < num_threads; i++) {
        sum += H_out[i];
        printf("\n %i", H_out[i]);
    }
    printf("\n Time Sum Avg Avgt/elem %f %li %li %f", tm, sum, sum / num_threads,
            sum / num_threads / tm);
    printf("\n");
}
