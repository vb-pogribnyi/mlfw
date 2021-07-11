#include "pch.h"
#include "common.h"
#include <operation/Conv.h>

TEST(Conv, Instantiation) {
	Conv1d conv(1, 1, 1);
	EXPECT_EQ(true, true);
}

TEST(Conv, Run_1) {
	float weight_data[] = { 1 };
	Conv1d conv(1, 1, 1);
	conv.weight->setData(weight_data);
	conv.bias->setData(weight_data);

	float data[] = { 3 };
	float exp_output[] = { 4 };
	Tensor input(vector<int>({ 1, 1, 1 }), data);
	Tensor output(vector<int>({ 1, 1, 1 }));
	conv.run(&input, &output);
	vector<float> output_data = output.getData();
	Tensor::sync();
	compare_arrays(exp_output, &output_data[0], output_data.size());
	Tensor::reset();
}

TEST(Conv, Run_2) {
	Conv1d conv(1, 1, 3);

	Tensor input(vector<int>({ 1, 1, 1 }));
	Tensor output(vector<int>({ 1, 1, 1 }));
	EXPECT_THROW(conv.run(&input, &output), TensorShapeError);
}

TEST(Conv, Run_3) {
	float weight_data[] = { 1, 0.5 };
	float bias_data[] = { 1 };
	Conv1d conv(1, 2, 1);
	conv.weight->setData(weight_data);
	conv.bias->setData(weight_data);

	float data[] = { 2 };
	float exp_output[] = { 3, 2 };
	Tensor input(vector<int>({ 1, 1, 1 }), data);
	Tensor output(vector<int>({ 1, 2, 1 }));
	conv.run(&input, &output);
	vector<float> output_data = output.getData();
	Tensor::sync();
	compare_arrays(exp_output, &output_data[0], output_data.size());
	Tensor::reset();
}
