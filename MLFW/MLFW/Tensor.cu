#include "Tensor.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

Tensor::Tensor(vector<int> shape, float* data) : shape(shape) {
	//
}
