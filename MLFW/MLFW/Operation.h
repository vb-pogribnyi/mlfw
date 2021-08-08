#pragma once
#include "Tensor.h"

class Operation
{
public:
	virtual void run(Tensor* output, Tensor* input1, Tensor* input2 = 0) = 0;
	// Update internal weights (if any) with learning rate lr
	virtual void update(float lr) = 0;
	// Propagate gradients backwards using sensitivity
	virtual void propagate() = 0;
protected:
	void record_flow(Tensor* output, Tensor* input1, Tensor* input2 = 0);
	Tensor* flow_input1 = 0;
	Tensor* flow_input2 = 0;
	Tensor* flow_output = 0;
};
