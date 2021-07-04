#include "Conv.h"
#include "../Common.cuh"
#include <iostream>

using namespace std;
extern cudaError_t cudaStatus;


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

void Conv1d::run(Tensor* input, Tensor* output) {
	//
}

void Conv1d::update(float lr) {
	//
}

void Conv1d::propagate(Tensor* input, Tensor* output) {
	//
}
