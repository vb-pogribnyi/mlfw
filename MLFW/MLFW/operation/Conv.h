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

struct ConvOffset {
	int example, x_in, x_out, y_in, y_out, ch_in, ch_out;
};

struct ConvInfo {
	int in_idx, out_idx, kern_idx, bias_idx, out_width, out_height, n_examples, n_channels_out, n_channels_in;
};

class Conv1d : public Operation
{
public:
	Conv1d(const int ch_in, const int ch_out, const int width);
	~Conv1d();
	Tensor* weight;
	Tensor* bias;
	ConvOffset limits;

	void run(Tensor* output, Tensor* input, Tensor* _ = 0);
	void update(float lr);
	void propagate();
private:
	void checkShapes(vector<int> input_shape, vector<int> output_shape, vector<int> weight_shape);
};
