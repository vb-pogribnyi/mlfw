#include "SGD.h"
#include "../Common.cuh"
#include <iostream>

using namespace std;
extern cudaError_t cudaStatus;
#define SGD_PRINT_DEBUG false

SGD::SGD(list<Tensor*> tensors, float lr) : Optimizer(tensors, lr) {}

__global__ void sgd_step(CUDATensor* target, CUDATensor* grads, float lr) {
	target->data[blockIdx.x] -= lr * grads->data[blockIdx.x];
#if SGD_PRINT_DEBUG
	printf("Idx: %i, data: %2.5f, grad: %2.3f\n", blockIdx.x, target->data[blockIdx.x], grads->data[blockIdx.x]);
#endif
}

void SGD::step() {
	for (Tensor* t : tensors) {
		sgd_step << <t->getSize(), 1 >> > (t->getCudaData(), t->getCudaGrad(), lr);
		HE(cudaPeekAtLastError());
	}
}
