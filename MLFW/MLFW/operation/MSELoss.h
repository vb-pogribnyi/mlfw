#pragma once

#include "../Operation.h"
#include <vector>

using namespace std;

class MSELoss : public Operation
{
public:
	MSELoss(/* Loss Weights */);
	~MSELoss();

	void run(Tensor* output, Tensor* input, Tensor* target);
	void update(float lr);
	void propagate(Tensor* input, Tensor* output);
};
