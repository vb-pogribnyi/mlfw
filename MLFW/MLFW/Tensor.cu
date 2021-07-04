#include "Tensor.h"
#include "Common.cuh"
#include <iostream>

using namespace std;
extern cudaError_t cudaStatus;

Tensor::Tensor(vector<int> shape, float* data) : shape(shape) {
	// Calculate size & dims
	size = 1;
	dims = 0;
	for (int s : shape) {
		size *= s;
		dims += s;
	}
	// Copy data
	this->data.reserve(size);
	for (int i = 0; i < size; i++) {
		this->data.push_back(data[i]);
	}
	// Upload data onto GPU
	CUDATensor d_data = createCudaTensor();
	HE(cudaMemcpy(d_data.data, &(data[0]), size * sizeof(float), cudaMemcpyHostToDevice));
	HE(cudaMalloc((void**)&(cuda_data), sizeof(CUDATensor)));
	HE(cudaMalloc((void**)&(cuda_grad), sizeof(CUDATensor)));
	HE(cudaMalloc((void**)&(cuda_sens), sizeof(CUDATensor)));
	HE(cudaMemcpy(cuda_data, &d_data, sizeof(CUDATensor), cudaMemcpyHostToDevice));
	HE(cudaMemcpy(cuda_grad, &createCudaTensor(), sizeof(CUDATensor), cudaMemcpyHostToDevice));
	HE(cudaMemcpy(cuda_sens, &createCudaTensor(), sizeof(CUDATensor), cudaMemcpyHostToDevice));
}

CUDATensor Tensor::createCudaTensor() {
	CUDATensor result;
	HE(cudaMalloc((void**)&(result.data), size * sizeof(float)));
	HE(cudaMemset(result.data, 0, size * sizeof(float)));
	HE(cudaMalloc((void**)&(result.shape), shape.size() * sizeof(int)));
	HE(cudaMemcpy(result.shape, &(shape[0]), shape.size() * sizeof(int), cudaMemcpyHostToDevice));
	result.dims = dims;
	result.size = size;
	return result;
}

void Tensor::loadData(CUDATensor* src) {
	CUDATensor temp;
	HE(cudaMemcpy(&temp, src, sizeof(CUDATensor), cudaMemcpyDeviceToHost));
	HE(cudaMemcpy(&data[0], temp.data, size * sizeof(float), cudaMemcpyDeviceToHost));
}

vector<float> Tensor::getData() {
	loadData(cuda_data);
	return vector<float>(data);
}
vector<float> Tensor::getGrad() {
	loadData(cuda_grad);
	return vector<float>(data);
}

vector<float> Tensor::getSens() {
	loadData(cuda_sens);
	return vector<float>(data);
}	

void Tensor::sync() {
	HE(cudaDeviceSynchronize());
}

void Tensor::reset() {
	HE(cudaDeviceReset());
}
