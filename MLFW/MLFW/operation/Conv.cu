#include "Conv.h"
#include "../Common.cuh"
#include <iostream>

#define RUN_CONV(function, offset, limits, weight_shape, output_shape, parameters) \
	for (offset.example = 0; offset.example < output_shape[0]; offset.example += limits.example) { \
		for (offset.ch_in = 0; offset.ch_in < weight_shape[0]; offset.ch_in += limits.ch_in) { \
			for (offset.ch_out = 0; offset.ch_out < weight_shape[1]; offset.ch_out += limits.ch_out) { \
				for (offset.x_out = 0; offset.x_out < output_shape[2]; offset.x_out += limits.x_out) { \
					for (offset.x_in = 0; offset.x_in < weight_shape[2]; offset.x_in += limits.x_in) { \
						int grid_x = min(output_shape[2] - offset.x_out, limits.x_out); \
						int grid_y = 1; \
						int grid_z = min(output_shape[0] - offset.example, limits.example); \
						dim3 grid(grid_x, grid_y, grid_z); \
						int block_x = min(weight_shape[2] - offset.x_in, limits.x_in); \
						int block_y = 1; \
						int block_z = min(weight_shape[0] - offset.ch_in, limits.ch_in) * \
							min(weight_shape[1] - offset.ch_out, limits.ch_out); \
						dim3 block(block_x, block_y, block_z); \
						HE(cudaMemcpy(d_offset, &offset, sizeof(ConvOffset), cudaMemcpyHostToDevice)); \
						function <<<grid, block>>> parameters; \
						HE(cudaPeekAtLastError()); \
					} \
				} \
			} \
		} \
	} \

#define	CONV_PRINT_DEBUG false
#define CONV_BACK_PRINT_DEBUG false

using namespace std;
extern cudaError_t cudaStatus;

__device__ ConvInfo get_indices(CUDATensor* input, CUDATensor* output, CUDATensor* weight, ConvOffset* offset) {
	ConvInfo result = { 0 };
	int in_width = 1;
	int in_height = 1;
	result.out_width = 1;
	result.out_height = 1;
	result.n_examples = input->shape[0];
	result.n_channels_in = weight->shape[0];
	result.n_channels_out = weight->shape[1];
	if (input->dims == 4) {
		in_width = input->shape[input->dims - 2];
		in_height = input->shape[input->dims - 1];
		result.out_width = output->shape[output->dims - 2];
		result.out_height = output->shape[output->dims - 1];
	}
	else {
		in_width = input->shape[input->dims - 1];
		result.out_width = output->shape[output->dims - 1];
	}
	int x_out = offset->x_out + blockIdx.x;
	int y_out = offset->y_out + blockIdx.y;
	int x_in = offset->x_in + threadIdx.x;
	int y_in = offset->y_in + threadIdx.y;

	int channel = offset->ch_in * result.n_channels_out + offset->ch_out + threadIdx.z;
	int ch_in = channel / result.n_channels_out;
	int ch_out = channel % result.n_channels_out;
	int example = offset->example + blockIdx.z;

	result.in_idx = example * result.n_channels_in * in_width * in_height +						// example
		ch_in * in_height * in_width +															// in channel
		(y_out + y_in) * in_width +																// height
		x_out + x_in;																			// width

	result.kern_idx = ch_in * weight->shape[2] * weight->shape[3] +								// in channel
		ch_out * weight->shape[0] * weight->shape[2] * weight->shape[3] +						// out channel
		y_in * weight->shape[3] +																// height
		x_in;																					// width

	result.out_idx = example * result.n_channels_out * result.out_width * result.out_height +	// example
		ch_out * result.out_width * result.out_height +											// out channel
		y_out * result.out_width + 																// height
		x_out;																					// width
	result.bias_idx = ch_out;

	return result;
}

__global__ void convolve(CUDATensor* input, CUDATensor* output, CUDATensor* weight, CUDATensor* bias, ConvOffset* offset) {
	ConvInfo indices = get_indices(input, output, weight, offset);

	atomicAdd(output->data + indices.out_idx, input->data[indices.in_idx] * weight->data[indices.kern_idx]);

#if CONV_PRINT_DEBUG
	printf("Thread idx: %i, offset_in: %i, offset_out: %i, channel: %i\n",
		threadIdx.z, offset->ch_in, offset->ch_out, channel);
	printf("example: %i, ch_out: %i, ch_in: %i, in_idx: %i, kern_idx: %i, bias_idx: %i, out_idx: %i\n",
		example, ch_out, ch_in, in_idx, kern_idx, bias_idx, out_idx);
	printf("Output: %2.5f\n",
		output->data[out_idx]);
#endif
}

__global__ void add_bias(CUDATensor* input, CUDATensor* output, CUDATensor* weight, CUDATensor* bias, ConvOffset* offset) {
	ConvInfo indices = get_indices(input, output, weight, offset);

	atomicAdd(output->data + indices.out_idx, bias->data[indices.bias_idx]);
#if CONV_PRINT_DEBUG
	printf("BIAS:\n");
	printf("Out idx: %i, bias idx: %i, output: %2.3f\n", out_idx, bias_idx, output->data[out_idx]);
#endif
}

__global__ void convolve_backward(CUDATensor* input, CUDATensor* d_input, CUDATensor* d_output,
	CUDATensor* weight, CUDATensor* bias, 
	CUDATensor* d_weight, CUDATensor* d_bias, 
	ConvOffset* offset) {
	ConvInfo indices = get_indices(input, d_output, weight, offset);
	int n_vals_w = indices.out_width * indices.out_height * indices.n_examples * indices.n_channels_out;
	int n_vals_b = indices.out_width * indices.out_height * indices.n_examples * indices.n_channels_out * 
		indices.n_channels_in * weight->shape[2] * weight->shape[3];

	atomicAdd(d_weight->data + indices.kern_idx, input->data[indices.in_idx] * d_output->data[indices.out_idx] / n_vals_w);
	atomicAdd(d_bias->data + indices.bias_idx, d_output->data[indices.out_idx] / n_vals_b);

	atomicAdd(d_input->data + indices.in_idx, d_output->data[indices.out_idx] * weight->data[indices.kern_idx] / n_vals_w);

#if CONV_BACK_PRINT_DEBUG
		printf("Width: %i, height: %i, in idx: %i, out idx: %i, w_idx: %i, b_idx: %i\n", in_width, in_height, in_idx, out_idx, w_idx, b_idx);
		printf("Input: %2.3f, output grad: %2.3f, values affected by W: %i, values affected by B: %i\n", input->data[in_idx], d_output->data[out_idx], n_vals_w, n_vals_b);
		printf("New dw: %2.5f, new db: %2.5f, w_idx: %i, b_idx: %i\n", d_weight->data[w_idx], d_bias->data[b_idx], w_idx, b_idx);
		printf("Sens: %2.5f, weight: %2.3f, grad: %2.3f, in_idx: %i, out_idx: %i, w_idx: %i\n", d_input->data[in_idx], weight->data[w_idx], d_output->data[out_idx], in_idx, out_idx, w_idx);
#endif
}


Conv1d::Conv1d(const int ch_in, const int ch_out, const int width) : weight(0), bias(0) {
	limits.ch_in = 1;
	limits.ch_out = 1;
	limits.example = 1;
	limits.x_in = 1;
	limits.y_in = 1;
	limits.x_out = 1;
	limits.y_out = 1;

	limits.ch_in = 2;
	limits.ch_out = 2;
	limits.example = 16;
	limits.x_in = 8;
	limits.y_in = 8;
	limits.x_out = 16;
	limits.y_out = 16;

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
	int exp_out_width = input_shape[2] - 2 * (weight_shape[2] / 2);
	int exp_out_channels = weight_shape[1];
	int exp_in_channels = weight_shape[0];
	if (input_shape[1] != exp_in_channels)
		throw TensorShapeError();
	if (output_shape[1] != exp_out_channels)
		throw TensorShapeError();
	if (exp_out_width <= 0 || output_shape[2] != exp_out_width)
		throw TensorShapeError();

}

void Conv1d::run(Tensor* output, Tensor* input, Tensor* _) {
	record_flow(output, input);
	output->clear();
	vector<int> input_shape = input->getShape();
	vector<int> output_shape = output->getShape();
	vector<int> weight_shape = weight->getShape();
	// throws TensorShapeError
	checkShapes(input_shape, output_shape, weight_shape);
	ConvOffset offset = { 0 };
	ConvOffset* d_offset;
	HE(cudaMalloc((void**)&(d_offset), sizeof(ConvOffset)));

	RUN_CONV(convolve, offset, limits, weight_shape, output_shape, (
		input->getCudaData(),
		output->getCudaData(),
		weight->getCudaData(),
		bias->getCudaData(),
		d_offset
		)
	);
	auto limits_local = limits;
	limits_local.x_in = 1;
	limits_local.ch_in = 1;
	weight_shape[0] = 1;
	weight_shape[2] = 1;
	RUN_CONV(add_bias, offset, limits_local, weight_shape, output_shape, (
		input->getCudaData(),
		output->getCudaData(),
		weight->getCudaData(),
		bias->getCudaData(),
		d_offset
		)
	);
}

void Conv1d::update(float lr) {
	//
}

void Conv1d::propagate() {
	// Out = f(Wx + b)
	// dOut/dW = df/d(Wx + b) * x

	weight->clear(true);
	bias->clear(true);
	vector<int> input_shape = flow_input1->getShape();
	vector<int> output_shape = flow_output->getShape();
	vector<int> weight_shape = weight->getShape();

	ConvOffset offset = { 0 };
	ConvOffset* d_offset;

	HE(cudaMalloc((void**)&(d_offset), sizeof(ConvOffset)));

	RUN_CONV(convolve_backward, offset, limits, weight_shape, output_shape, (
		flow_input1->getCudaData(),
		flow_input1->getCudaGrad(),
		flow_output->getCudaGrad(),
		weight->getCudaData(),
		bias->getCudaData(),
		weight->getCudaGrad(),
		bias->getCudaGrad(),
		d_offset)
	);
}
