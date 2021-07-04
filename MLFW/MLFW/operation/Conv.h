#pragma once

#include "../Operation.h"
#include <vector>

using namespace std;

class Conv1d : public Operation
{
public:
	Conv1d(const int ch_in, const int ch_out, const int width);
	~Conv1d();
	Tensor* weight;
	Tensor* bias;

	void run(Tensor* input, Tensor* output);
	void update(float lr);
	void propagate(Tensor* input, Tensor* output);
};
