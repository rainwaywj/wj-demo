/*
 * @Author: wujun
 * @Date: 2019-05-20 20:26:04
 * @Last Modified by:   wujun
 * @Last Modified time: 2019-05-20 20:26:04
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/timeb.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>
#include "cuda_runtime.h"

// #include "opencv2/core/core.hpp"
// #include "opencv2/opencv.hpp"
// #include "opencv2/imgproc/imgproc.hpp"


void test_cuda_malloc(size_t memory_size) {
    void* ptr = nullptr;
    auto  err = cudaMalloc(&ptr, memory_size);
    if (0 != err) {
        printf("cudaMalloc error: %d\n", err);
        fflush(stdout);
        return;
    }
    printf("press any key to exit!");
    fflush(stdout);
    getchar();
    err = cudaFree(ptr);
    if (0 != err) {
        printf("cudaFree error: %d\n", err);
        fflush(stdout);
    }
    return;
}

int main(int argc, char** argv) {
    auto gpu_id = atoi(argv[1]);
    printf("input gpu id: %d\n", gpu_id);
    auto memory_size = atol(argv[2]);
    printf("input memory size: %ld\n", memory_size);

    auto err = cudaSetDevice(gpu_id);
    if (0 != err) {
        printf("cudaSetDevice error: %d\n", err);
        fflush(stdout);
        return -1;
    }

    test_cuda_malloc(memory_size);
    printf("[%s,%d,%s] info: Demo test_cuda is Completed!_#_!\n", __FILE__,
           __LINE__, __FUNCTION__);

    return 0;
}