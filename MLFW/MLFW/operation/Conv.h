#pragma once

#include "../Operation.h"
#include <vector>

using namespace std;

class TensorShapeError : public exception
{
	const char* what() const throw ()
	{
		return "Wrong tensor shape";
	}
};

class Conv1d : public Operation
{
public:
	Conv1d(const int ch_in, const int ch_out, const int width);
	~Conv1d();
	Tensor* weight;
	Tensor* bias;

	void run(Tensor* output, Tensor* input, Tensor* _ = 0);
	void update(float lr);
	void propagate(Tensor* input, Tensor* output);
private:
	void checkShapes(vector<int> input_shape, vector<int> output_shape, vector<int> weight_shape);
};
