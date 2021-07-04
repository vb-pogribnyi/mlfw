#ifndef COMMON_H
#define COMMON_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define HE(function) \
	cudaStatus = function; \
	if (cudaStatus != cudaSuccess) { \
		cout << __FILE__ << ' ' << __LINE__ << endl; \
		cout << cudaGetErrorName(cudaStatus) << endl; \
		cout << cudaGetErrorString(cudaStatus) << endl; \
	}

#endif
