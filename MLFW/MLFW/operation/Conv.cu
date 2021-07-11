﻿#include "Conv.h"
#include "../Common.cuh"
#include <iostream>

#define	PRINT_DEBUG true

using namespace std;
extern cudaError_t cudaStatus;

__global__ void convolve(CUDATensor* input, CUDATensor* output, CUDATensor* weight, CUDATensor* bias) {
	extern __shared__ float s[];
	for (int ch_out = 0; ch_out < output->shape[1]; ch_out++) {
		int in_idx = blockIdx.z * gridDim.y * gridDim.x +											// example
			threadIdx.z * weight->shape[1] * weight->shape[2] * weight->shape[3] +					// in channel
			(blockIdx.y /*+ threadIdx.y offset*/) * gridDim.x +										// height
			(blockIdx.x /*+ threadIdx.x offset*/);													// width

		int kern_idx = threadIdx.z * weight->shape[1] * weight->shape[2] * weight->shape[3] +		// in channel
			ch_out * weight->shape[2] * weight->shape[3] +											// out channel
			threadIdx.y * weight->shape[3] +														// height
			threadIdx.x;																			// width

		int out_idx = blockIdx.z * gridDim.y * gridDim.x + 											// example
			ch_out * weight->shape[2] * weight->shape[3] +											// out channel
			blockIdx.y * gridDim.x + 																// height
			blockIdx.x;																				// width

		int shared_idx = threadIdx.z * weight->shape[1] * weight->shape[2] * weight->shape[3] +		// in channel
			threadIdx.y * weight->shape[3] +														// height
			threadIdx.x;																			// width
		int bias_idx = threadIdx.z;
#if PRINT_DEBUG
		printf("Channel: %i, in_idx: %i, kern_idx: %i, bias_idx: %i, out_idx: %i\n", ch_out, in_idx, kern_idx, bias_idx, out_idx);
		printf("Weight: in %i out %i w %i h %i\n", weight->shape[0], weight->shape[1], weight->shape[2], weight->shape[3]);
#endif
		s[shared_idx] = input->data[in_idx] * weight->data[kern_idx] + bias->data[bias_idx];
#if PRINT_DEBUG
		printf("In: %2.3f, Kern: %2.3f, Bias: %2.3f, Output: %2.3f\n", input->data[in_idx], weight->data[kern_idx], bias->data[bias_idx], s[shared_idx]);
#endif
		__syncthreads();

		if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
			for (int i = 1; i < blockDim.x * blockDim.y * blockDim.z; i++) {
				printf("Sum idx: %i, output: %2.3f\n", i, s[i]);
				s[0] += s[i];
			}
			output->data[out_idx] = s[0];
		}
		__syncthreads();
#if PRINT_DEBUG
		printf("Out idx: %i, output: %2.3f\n", out_idx, s[0]);
		printf("\n");
#endif
	}
}


Conv1d::Conv1d(const int ch_in, const int ch_out, const int width) : weight(0), bias(0) {
	vector<int> weight_shape = { ch_in, ch_out, width, 1 };
	vector<int> bias_shape = { ch_out };
	vector<float> weight_data(ch_in * ch_out * width, 0);
	vector<float> bias_data(ch_out, 0);

	weight = new Tensor(weight_shape, &weight_data[0]);
	bias = new Tensor(bias_shape, &bias_data[0]);
}

Conv1d::~Conv1d() {
	if (weight) {
		delete weight;
	}
	if (bias) {
		delete bias;
	}
}

void Conv1d::checkShapes(vector<int> input_shape, vector<int> output_shape, vector<int> weight_shape) {
	int exp_out_width = input_shape[2] - weight_shape[2] / 2;
	int exp_out_channels = weight_shape[1];
	int exp_in_channels = weight_shape[0];
	if (input_shape[1] != exp_in_channels)
		throw TensorShapeError();
	if (output_shape[1] != exp_out_channels)
		throw TensorShapeError();
	if (exp_out_width <= 0 || output_shape[2] != exp_out_width)
		throw TensorShapeError();

}

void Conv1d::run(Tensor* input, Tensor* output) {
	vector<int> input_shape = input->getShape();
	vector<int> output_shape = output->getShape();
	vector<int> weight_shape = weight->getShape();
	// throws TensorShapeError
	checkShapes(input_shape, output_shape, weight_shape);
	// grid: width x height x examples
	dim3 grid(output_shape[2], 1, output_shape[0]);
	// block: kernel_width x kernel_height x in_channels
	dim3 block(weight_shape[2], 1, weight_shape[0]);
	int shared_mem_items = weight_shape[0] * weight_shape[2] * weight_shape[3];

	convolve << <grid, block, sizeof(float) * shared_mem_items >> > (input->getCudaData(),
		output->getCudaData(),
		weight->getCudaData(),
		bias->getCudaData());
	HE(cudaPeekAtLastError());
}

void Conv1d::update(float lr) {
	//
}

void Conv1d::propagate(Tensor* input, Tensor* output) {
	//
}
