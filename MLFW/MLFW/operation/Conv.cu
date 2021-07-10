#include "Conv.h"
#include "../Common.cuh"
#include <iostream>

using namespace std;
extern cudaError_t cudaStatus;

__global__ void convolve(CUDATensor* input, CUDATensor* output, CUDATensor* weight, CUDATensor* bias) {
	output->data[0] = 3;
}


Conv1d::Conv1d(const int ch_in, const int ch_out, const int width) : weight(0), bias(0) {
	vector<int> weight_shape = { ch_in, ch_out, width };
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
	dim3 grid(input_shape[2], 1, input_shape[0]);
	// block: kernel_width x kernel_height x in_channels
	dim3 block(weight_shape[2], 1, weight_shape[0]);

	convolve << <grid, block, weight->getSize() >> > (input->getCudaData(),
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
