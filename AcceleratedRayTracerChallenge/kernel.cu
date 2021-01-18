#include "Vec4.cuh"

#include "thrust/host_vector.h"
#include "thrust/device_vector.h"


__global__ void render_kernel(int width, int height)
{

    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;

    if ((i >= width) || (j >= height)) return;
    const int pixel_index = j * width + i;


    RayTracer::Vec4 t = RayTracer::Vec4(0, 0, 0) + RayTracer::Vec4(1, 1, 1);
}

void run() {

    dim3 blocks(300 / 9, 300 / 9);
    dim3 threads(8, 8);

    render_kernel  << < blocks, threads >> > (300,300);
    cudaDeviceSynchronize();
}

int main()
{


    run();
    return 0;
}

