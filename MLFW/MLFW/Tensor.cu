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
		dims += 1;
	}
	// Copy data
	this->data.reserve(size);
	for (int i = 0; i < size; i++) {
		float value = data == 0 ? 0 : data[i];
		this->data.push_back(value);
	}
	// Upload data onto GPU
	CUDATensor d_data = createCudaTensor();
	HE(cudaMemcpy(d_data.data, &(this->data[0]), size * sizeof(float), cudaMemcpyHostToDevice));
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
	// TODO: set 1 for the initial grad data
	// TODO: get rid of sensitivity tensor
	HE(cudaMemset(result.data, 0, size * sizeof(float)));
	HE(cudaMalloc((void**)&(result.shape), shape.size() * sizeof(int)));
	HE(cudaMemcpy(result.shape, &(shape[0]), shape.size() * sizeof(int), cudaMemcpyHostToDevice));
	result.dims = dims;
	result.size = size;
	return result;
}

void Tensor::downloadData(CUDATensor* src) {
	CUDATensor temp;
	HE(cudaMemcpy(&temp, src, sizeof(CUDATensor), cudaMemcpyDeviceToHost));
	HE(cudaMemcpy(&data[0], temp.data, size * sizeof(float), cudaMemcpyDeviceToHost));
}

void Tensor::uploadData(float* data, CUDATensor* dst) {
	CUDATensor temp;
	HE(cudaMemcpy(&temp, dst, sizeof(CUDATensor), cudaMemcpyDeviceToHost));
	HE(cudaMemcpy(temp.data, data, size * sizeof(float), cudaMemcpyHostToDevice));
}

vector<float> Tensor::getData() {
	downloadData(cuda_data);
	return vector<float>(data);
}
vector<float> Tensor::getGrad() {
	downloadData(cuda_grad);
	return vector<float>(data);
}

vector<float> Tensor::getSens() {
	downloadData(cuda_sens);
	return vector<float>(data);
}

void Tensor::clear(bool only_grad)
{
	CUDATensor temp;
	if (!only_grad) {
		HE(cudaMemcpy(&temp, cuda_data, sizeof(CUDATensor), cudaMemcpyDeviceToHost));
		HE(cudaMemset(temp.data, 0, size * sizeof(float)));
	}
	HE(cudaMemcpy(&temp, cuda_grad, sizeof(CUDATensor), cudaMemcpyDeviceToHost));
	HE(cudaMemset(temp.data, 0, size * sizeof(float)));
}

vector<int> Tensor::getShape() {
	return shape;
}

int Tensor::getSize() {
	return size;
}

void Tensor::sync() {
	HE(cudaDeviceSynchronize());
}

void Tensor::reset() {
	HE(cudaDeviceReset());
}

void Tensor::setData(float* data) {
	uploadData(data, cuda_data);
}

void Tensor::setGrad(float* data) {
	uploadData(data, cuda_grad);
}

CUDATensor* Tensor::getCudaData() {
	return cuda_data;
}

CUDATensor* Tensor::getCudaGrad() {
	return cuda_grad;
}

CUDATensor* Tensor::getCudaSens() {
	return cuda_sens;
}
