#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"

#include <cstdio>
#include <cstdint>
#include <cstdlib>

extern "C" __device__ void ptxTestFunc(int* dst, const int* src);

static __global__ void ptxTestKernel(int* dst, const int* src)
{
    ptxTestFunc(dst, src);
}

int main(void)
{
    cudaDeviceProp props{ };
    auto cudaStatus = cudaGetDeviceProperties(&props, 0);
    if (cudaStatus != cudaSuccess)
    {
        printf("cudaGetDeviceProperties failed: %s\n", cudaGetErrorString(cudaStatus));
        return 0;
    }

    auto const maxBlocksPerSM = props.maxBlocksPerMultiProcessor;
    auto const maxThreadsPerBlock = props.maxThreadsPerBlock;
    auto const maxThreadsPerSM = props.maxThreadsPerMultiProcessor;
    auto const regsPerSM = props.regsPerMultiprocessor;
    auto const sharedMemSizePerBlock = props.sharedMemPerBlock;
    auto const constMemSize = props.totalConstMem;

    printf("Current GPU: %s\n", props.name);
    printf("max blocks per SM: %d\n", maxBlocksPerSM);
    printf("max threads per block: %d\n", maxThreadsPerBlock);
    printf("max threads per SM: %d\n", maxThreadsPerSM);
    printf("registers per SM: %d\n", regsPerSM);
    printf("shared memroy size per block: %zuKB\n", sharedMemSizePerBlock / 1024);
    printf("constant memory size on the device: %zuKB\n", constMemSize / 1024);

    puts("\n======== ptxTestKernel kernel info ========");

    cudaFuncAttributes funcAttrs{ };
    cudaStatus = cudaFuncGetAttributes(&funcAttrs, ptxTestKernel);
    if (cudaStatus != cudaSuccess)
    {
        printf("cudaFuncGetAttributes failed: %s\n", cudaGetErrorString(cudaStatus));
        return 0;
    }
    printf("max threads per block: %d\n", funcAttrs.maxThreadsPerBlock);
    printf("number of registers by each thread: %d\n", funcAttrs.numRegs);
    printf("local memory size by each thread: %zu bytes\n", funcAttrs.localSizeBytes);
    printf("shared memory size per block: %zu bytes\n", funcAttrs.sharedSizeBytes);
    printf("constant memory size: %zu bytes\n", funcAttrs.constSizeBytes);
    puts("");

    constexpr int elemCount = 4096;
    int* hostSrc = new int[elemCount];
    for (int i = 0; i < elemCount; i++)
        hostSrc[i] = i + 1;

    int* devDst = nullptr;
    int* devSrc = nullptr;

    constexpr auto bufferSize = elemCount * sizeof(*hostSrc);

    do
    {
        cudaStatus = cudaMalloc(&devDst, bufferSize);
        if (cudaStatus != cudaSuccess)
        {
            printf("cudaMalloc devDst failed: %s\n", cudaGetErrorString(cudaStatus));
            break;
        }

        cudaStatus = cudaMemcpy(devDst, hostSrc, bufferSize, cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess)
        {
            printf("cudaMemcpy to devDst failed: %s\n", cudaGetErrorString(cudaStatus));
            break;
        }

        cudaStatus = cudaMalloc(&devSrc, bufferSize);
        if (cudaStatus != cudaSuccess)
        {
            printf("cudaMalloc devSrc failed: %s\n", cudaGetErrorString(cudaStatus));
            break;
        }

        cudaStatus = cudaMemcpy(devSrc, hostSrc, bufferSize, cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess)
        {
            printf("cudaMemcpy to devSrc failed: %s\n", cudaGetErrorString(cudaStatus));
            break;
        }

        constexpr int threadsPerBlock = 256;
        constexpr auto nBlocks = elemCount / threadsPerBlock;

        ptxTestKernel <<< nBlocks, threadsPerBlock >>> (devDst, devSrc);

        cudaStatus = cudaMemcpy(hostSrc, devDst, bufferSize, cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess)
            printf("cudaMemcpy to hostSrc failed: %s\n", cudaGetErrorString(cudaStatus));

        // result verification
        bool success = true;
        for (int i = 0; i < elemCount; i++)
        {
            if (hostSrc[i] != (i + 1) * 2)
            {
                success = false;
                break;
            }
        }
        printf("Is equal? %s\n", success ? "YES" : "NO");

    } while (false);

    if (hostSrc != nullptr)
        delete[] hostSrc;

    if (devDst != nullptr)
        cudaFree(devDst);
    if (devSrc != nullptr)
        cudaFree(devSrc);

    return 0;
}

