#pragma once
#include "Tensor.h"

class Operation
{
public:
	virtual void run(Tensor* output, Tensor* input1, Tensor* input2 = 0) = 0;
	// Update internal weights (if any) with learning rate lr
	virtual void update(float lr) = 0;
	// Propagate gradients backwards using sensitivity
	virtual void propagate(Tensor* input, Tensor* output) = 0;
};
