#include "Conv.h"
#include "../Common.cuh"
#include <iostream>

#define	CONV_PRINT_DEBUG false
#define CONV_BACK_PRINT_DEBUG true

using namespace std;
extern cudaError_t cudaStatus;

__global__ void convolve(CUDATensor* input, CUDATensor* output, CUDATensor* weight, CUDATensor* bias) {
	extern __shared__ float s[];
	for (int ch_out = 0; ch_out < output->shape[1]; ch_out++) {
		int in_width = 2 * (weight->shape[2] / 2) + gridDim.x;
		int in_height = 2 * (weight->shape[3] / 2) + gridDim.y;
		int in_idx = blockIdx.z * blockDim.z * in_width * in_height +					// example
			threadIdx.z * in_height * in_width +										// in channel
			(blockIdx.y + threadIdx.y) * gridDim.x +									// height
			(blockIdx.x + threadIdx.x);													// width

		int kern_idx = threadIdx.z * weight->shape[2] * weight->shape[3] +				// in channel
			ch_out * weight->shape[0] * weight->shape[2] * weight->shape[3] +			// out channel
			threadIdx.y * weight->shape[3] +											// height
			threadIdx.x;																// width

		int out_idx = blockIdx.z * weight->shape[1] * gridDim.y * gridDim.x + 			// example
			//ch_out * weight->shape[2] * weight->shape[3] +							// out channel
			ch_out * gridDim.y * gridDim.x +											// out channel
			blockIdx.y * gridDim.x + 													// height
			blockIdx.x;																	// width

		int shared_idx = threadIdx.z * weight->shape[2] * weight->shape[3] +			// in channel
			threadIdx.y * weight->shape[3] +											// height
			threadIdx.x;																// width
		int bias_idx = ch_out;
#if CONV_PRINT_DEBUG
		printf("example: %i, ch_out: %i, ch_in: %i, in_idx: %i, kern_idx: %i, bias_idx: %i, out_idx: %i, shared_idx: %i\n", 
			blockIdx.z, ch_out, threadIdx.z, in_idx, kern_idx, bias_idx, out_idx, shared_idx);
		printf("Weight: in %i out %i w %i h %i\n", weight->shape[0], weight->shape[1], weight->shape[2], weight->shape[3]);
#endif
		s[shared_idx] = input->data[in_idx] * weight->data[kern_idx];
#if CONV_PRINT_DEBUG
		printf("In: %2.3f, Kern: %2.3f, Bias: %2.3f, Output: %2.3f\n", input->data[in_idx], weight->data[kern_idx], bias->data[bias_idx], s[shared_idx]);
#endif
		__syncthreads();

		if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
#if CONV_PRINT_DEBUG
			printf("Sum idx: %i, output: %2.3f\n", 0, s[0]);
#endif
			for (int i = 1; i < blockDim.x * blockDim.y * blockDim.z; i++) {
#if CONV_PRINT_DEBUG
				printf("Sum idx: %i, output: %2.3f\n", i, s[i]);
#endif
				s[0] += s[i];
			}
			output->data[out_idx] = s[0] + bias->data[bias_idx];
#if CONV_PRINT_DEBUG
			printf("\n");
			printf("Out idx: %i, output: %2.3f\n", out_idx, s[0]);
#endif
		}
		__syncthreads();
#if CONV_PRINT_DEBUG
		printf("\n");
#endif
	}
}

__global__ void convolve_backward(CUDATensor* input, CUDATensor* d_input, CUDATensor* d_output,
	CUDATensor* weight, CUDATensor* bias, 
	CUDATensor* d_weight, CUDATensor* d_bias) {
	// Calculate input index
	// Calculate output index
	// Assign the gradient value to the corresponding weight
	// Divide the grad value by the number of values it has affected

//	dim3 grid(weight_shape[2], weight_shape[3], weight_shape[0] * weight_shape[1]);
//	dim3 block(output_shape[1], output_shape[2], output_shape[0]);
	int in_width = 2 * (weight->shape[2] / 2) + blockDim.x;
	int in_height = 2 * (weight->shape[3] / 2) + blockDim.y;
	int in_offset_w = blockIdx.x - (weight->shape[2] / 2);
	int in_offset_h = blockIdx.y - (weight->shape[3] / 2);
	// For backpropagation, in and out channels are reversed
	int n_channels_in = weight->shape[1];
	int n_channels_out = weight->shape[0];
	int curr_channel_in = blockIdx.z / n_channels_in;
	int curr_channel_out = blockIdx.z % n_channels_in;
	// Number of output values affected by singel weight or bias value
	int n_vals_w = blockDim.x * blockDim.y * blockDim.z;
	int n_vals_b = blockDim.x * blockDim.y * blockDim.z * n_channels_out * weight->shape[2] * weight->shape[3];

	int in_idx = threadIdx.z * in_width * in_height * n_channels_in +
		curr_channel_in * in_width * in_height +
		(threadIdx.y + in_offset_h + weight->shape[3] / 2) * in_width +
		(threadIdx.x + in_offset_w + weight->shape[2] / 2);
	int out_idx = threadIdx.z * blockDim.y * blockDim.x * n_channels_out +
		curr_channel_out * blockDim.y * blockDim.x +
		threadIdx.y * blockDim.x +
		threadIdx.x;
	int w_idx = curr_channel_in * weight->shape[1] * weight->shape[2] * weight->shape[3] +
		curr_channel_out * weight->shape[2] * weight->shape[3] +
		blockIdx.x * weight->shape[3] +
		blockIdx.y;
	int b_idx = curr_channel_out;

	//d_weight->data[w_idx] += 1;
	//d_bias->data[b_idx] += 1;
	atomicAdd(d_weight->data + w_idx, input->data[in_idx] * d_output->data[out_idx] / n_vals_w / n_channels_in);
	atomicAdd(d_bias->data + b_idx, d_output->data[out_idx] / n_vals_b / n_channels_in);

	atomicAdd(d_input->data + in_idx, d_output->data[out_idx] * weight->data[w_idx] / n_vals_w);

#if CONV_BACK_PRINT_DEBUG
	printf("Width: %i, height: %i, in idx: %i, out idx: %i, w_idx: %i, b_idx: %i\n", in_width, in_height, in_idx, out_idx, w_idx, b_idx);
	printf("Input: %2.3f, output grad: %2.3f, values affected by W: %i, values affected by B: %i\n", input->data[in_idx], d_output->data[out_idx], n_vals_w, n_vals_b);
	printf("New dw: %2.5f, new db: %2.5f, w_idx: %i, b_idx: %i\n", d_weight->data[w_idx], d_bias->data[b_idx], w_idx, b_idx);
#endif
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

	convolve << <grid, block, sizeof(float) * shared_mem_items >> > (
		input->getCudaData(),
		output->getCudaData(),
		weight->getCudaData(),
		bias->getCudaData());
	HE(cudaPeekAtLastError());
}

void Conv1d::update(float lr) {
	//
}

void Conv1d::propagate() {
	// Out = f(Wx + b)
	// dOut/dW = df/d(Wx + b) * x

	vector<int> input_shape = flow_input1->getShape();
	vector<int> output_shape = flow_output->getShape();
	vector<int> weight_shape = weight->getShape();
	// grid: kernel_width x kernel_height x (in_channels*out_channels)
	dim3 grid(weight_shape[2], weight_shape[3], weight_shape[0] * weight_shape[1]);
	// block: width-conv_padding x height-conv_padding x examples
	// A block for each convolution weight
	dim3 block(output_shape[2], 1, output_shape[0]);
	int shared_mem_items = weight_shape[0] * weight_shape[2] * weight_shape[3];

	convolve_backward << <grid, block, sizeof(float)* shared_mem_items >> > (
		flow_input1->getCudaData(),
		flow_input1->getCudaGrad(),
		flow_output->getCudaGrad(),
		weight->getCudaData(),
		bias->getCudaData(),
		weight->getCudaGrad(),
		bias->getCudaGrad());
	HE(cudaPeekAtLastError());
}
