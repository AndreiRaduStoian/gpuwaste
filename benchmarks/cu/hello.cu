#include <stdio.h>

__global__ void hello() {
    printf("hello from gpu\n");
}

int main() {
    hello<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
