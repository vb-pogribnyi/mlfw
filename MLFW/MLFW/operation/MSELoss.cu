#include "MSELoss.h"
#include "../Common.cuh"
#include <iostream>

#define	MSELOSS_PRINT_DEBUG false

using namespace std;
extern cudaError_t cudaStatus;

__global__ void mse_loss(CUDATensor* input, CUDATensor* target, CUDATensor* output) {
	extern __shared__ float s[];
	int shared_idx = threadIdx.y * blockDim.x + threadIdx.x;
	int n = blockDim.x * blockDim.y;
	int input_idx = blockIdx.x * n + shared_idx;
	int out_idx = blockIdx.x;
	s[shared_idx] = input->data[input_idx] - target->data[input_idx];
	if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
		s[0] *= s[0];
#if MSELOSS_PRINT_DEBUG
		printf("%2.3f\n", s[0]);
#endif
		for (int i = 1; i < n; i++) {
			s[0] += s[i] * s[i];
#if MSELOSS_PRINT_DEBUG
			printf("%2.3f\n", s[0]);
#endif
		}
		s[0] /= n;
	}


	output->data[out_idx] = s[0];

}


MSELoss::MSELoss() {
	// TODO: Initialize loss weights
}

MSELoss::~MSELoss() {
	//
}

void MSELoss::run(Tensor* output, Tensor* input, Tensor* target) {
	vector<int> input_shape = input->getShape();
	vector<int> output_shape = output->getShape();
	dim3 grid(input_shape[0], 1, 1);
	dim3 block(input_shape[1], input_shape[2], 1);
	int shared_mem_items = input_shape[1] + input_shape[2];
	mse_loss << <grid, block, sizeof(float) * shared_mem_items >> > (input->getCudaData(),
		target->getCudaData(),
		output->getCudaData());
	HE(cudaPeekAtLastError());
}

void MSELoss::update(float lr) {
	//
}

void MSELoss::propagate(Tensor* input, Tensor* output) {
	// No changes to weights
	// (a2(a1 * X + b1) + b2 - T)^2 -> 0
	// 2 * E * I2
	// 2 * E * a2 * X

	// Set input vector gradient to 2 * E, twice the output

}
