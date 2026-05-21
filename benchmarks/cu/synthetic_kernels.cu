#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdint.h>

#define CUDA_CHECK(x) do {                                      \
    cudaError_t err = (x);                                      \
    if (err != cudaSuccess) {                                   \
        printf("CUDA error %s:%d: %s\n",                        \
               __FILE__, __LINE__, cudaGetErrorString(err));    \
        exit(1);                                                \
    }                                                           \
} while (0)


-----------------------------------------------------------

__global__ void compute_fma_kernel(float *out, int iterations)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    float x = 1.001f + tid * 0.000001f;
    float y = 0.999f;
    float z = 0.5f;

    for (int i = 0; i < iterations; i++) {
        x = fmaf(x, y, z);
        y = fmaf(y, z, x);
        z = fmaf(z, x, y);
    }

    out[tid] = x + y + z;
}


-----------------------------------------------------------

__global__ void vector_add_kernel(
    const float *a,
    const float *b,
    float *c,
    int n
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}



__global__ void pointer_chase_kernel(
    const int *next,
    int *out,
    int n,
    int iterations
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= n) {
        return;
    }

    int idx = tid;

    for (int i = 0; i < iterations; i++) {
        idx = next[idx];
    }

    out[tid] = idx;
}



__global__ void shared_barrier_kernel(
    const float *in,
    float *out,
    int n,
    int iterations
)
{
    extern __shared__ float tile[];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local = threadIdx.x;

    float x = 0.0f;

    if (tid < n) {
        x = in[tid];
    }

    tile[local] = x;
    __syncthreads();

    for (int it = 0; it < iterations; it++) {
        float left  = tile[max(local - 1, 0)];
        float mid   = tile[local];
        float right = tile[min(local + 1, blockDim.x - 1)];

        __syncthreads();

        tile[local] = 0.25f * left + 0.5f * mid + 0.25f * right;

        __syncthreads();
    }

    if (tid < n) {
        out[tid] = tile[local];
    }
}


static void print_usage(const char *prog)
{
    printf("Usage:\n");
    printf("  %s <kernel> <n> <iterations> <block_size>\n\n", prog);
    printf("Kernels:\n");
    printf("  compute\n");
    printf("  vectoradd\n");
    printf("  pointer\n");
    printf("  shared\n\n");
    printf("Example:\n");
    printf("  %s compute 16777216 10000 256\n", prog);
}


static float run_timed_kernel(
    const char *kernel_name,
    int n,
    int iterations,
    int block_size
)
{
    int grid_size = (n + block_size - 1) / block_size;

    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_c = nullptr;
    int *d_next = nullptr;
    int *d_out_int = nullptr;

    CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, n * sizeof(float)));

    CUDA_CHECK(cudaMemset(d_a, 1, n * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_b, 2, n * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_c, 0, n * sizeof(float)));

    if (strcmp(kernel_name, "pointer") == 0) {
        int *h_next = (int *)malloc(n * sizeof(int));
        if (!h_next) {
            printf("Host allocation failed\n");
            exit(1);
        }

        for (int i = 0; i < n; i++) {
            h_next[i] = (i + 1) % n;
        }

        CUDA_CHECK(cudaMalloc(&d_next, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_out_int, n * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_next, h_next, n * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_out_int, 0, n * sizeof(int)));

        free(h_next);
    }

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));

    if (strcmp(kernel_name, "compute") == 0) {
        compute_fma_kernel<<<grid_size, block_size>>>(d_c, iterations);
    }
    else if (strcmp(kernel_name, "vectoradd") == 0) {
        vector_add_kernel<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
    }
    else if (strcmp(kernel_name, "pointer") == 0) {
        pointer_chase_kernel<<<grid_size, block_size>>>(d_next, d_out_int, n, iterations);
    }
    else if (strcmp(kernel_name, "shared") == 0) {
        size_t shmem_bytes = block_size * sizeof(float);
        shared_barrier_kernel<<<grid_size, block_size, shmem_bytes>>>(d_a, d_c, n, iterations);
    }
    else {
        printf("Unknown kernel: %s\n", kernel_name);
        print_usage("synthetic_benchmark");
        exit(1);
    }

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    if (d_next) CUDA_CHECK(cudaFree(d_next));
    if (d_out_int) CUDA_CHECK(cudaFree(d_out_int));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms;
}


int main(int argc, char **argv)
{
    if (argc != 5) {
        print_usage(argv[0]);
        return 1;
    }

    const char *kernel_name = argv[1];
    int n = atoi(argv[2]);
    int iterations = atoi(argv[3]);
    int block_size = atoi(argv[4]);

    if (n <= 0 || iterations <= 0 || block_size <= 0) {
        print_usage(argv[0]);
        return 1;
    }

    printf("Kernel:      %s\n", kernel_name);
    printf("N:           %d\n", n);
    printf("Iterations:  %d\n", iterations);
    printf("Block size:  %d\n", block_size);

    // wu
    run_timed_kernel(kernel_name, n, iterations, block_size);

    // timed
    float ms = run_timed_kernel(kernel_name, n, iterations, block_size);

    printf("Runtime_ms:  %.6f\n", ms);

    return 0;
}