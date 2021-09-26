#include "MSELoss.h"
#include "../Common.cuh"
#include <iostream>

#define	MSELOSS_PRINT_DEBUG false

using namespace std;
extern cudaError_t cudaStatus;

__global__ void mse_loss(CUDATensor* input, CUDATensor* target, CUDATensor* output) {
	int input_idx = blockIdx.z * gridDim.y * gridDim.x +
		blockIdx.y * gridDim.x +
		blockIdx.x;
	int n = gridDim.z * gridDim.y * gridDim.x;
	float err = target->data[input_idx] - input->data[input_idx];

	atomicAdd(output->data, err * err / n);
#if MSELOSS_PRINT_DEBUG
	printf("Output: %2.3f, N: %i\n", output->data[0], n);
#endif

}

__global__ void mse_loss_backward(CUDATensor* input, CUDATensor* target, CUDATensor* output) {

#if MSELOSS_PRINT_DEBUG
	printf("Backprop: %2.3f, %2.3f, %i\n", input->data[0], target->data[0], blockDim.y * blockDim.x);
#endif
	int idx = blockIdx.x * blockDim.y * blockDim.x +
		threadIdx.y * blockDim.x + threadIdx.x;

	//output->data[idx] = 2 * (target->data[idx] - input->data[idx]) / (blockDim.y * blockDim.x);
	output->data[idx] = -2 * (target->data[idx] - input->data[idx]) / (blockDim.x);
}

MSELoss::MSELoss() {
	// TODO: Initialize loss weights
}

MSELoss::~MSELoss() {
	//
}

void MSELoss::run(Tensor* output, Tensor* input, Tensor* target) {
	record_flow(output, input, target);
	output->clear();
	vector<int> input_shape = input->getShape();
	vector<int> output_shape = output->getShape();
	dim3 grid(input_shape[0], input_shape[1], input_shape[2]);
	dim3 block;
	mse_loss << <grid, block >> > (input->getCudaData(),
		target->getCudaData(),
		output->getCudaData());
	HE(cudaPeekAtLastError());
}

void MSELoss::update(float lr) {
	//
}

void MSELoss::propagate() {
	// No changes to weights
	// (a2(a1 * X + b1) + b2 - T)^2 -> 0
	// 2 * E * I2
	// 2 * E * a2 * X

	// Set input vector gradient to 2 * E, twice the output
	vector<int> input_shape = flow_input1->getShape();
	dim3 grid(input_shape[0], 1, 1);
	dim3 block(input_shape[1], input_shape[2], 1);
	mse_loss_backward << <grid, block >> > (flow_input1->getCudaData(),
		flow_input2->getCudaData(), 
		flow_input1->getCudaGrad());
	HE(cudaPeekAtLastError());
}
