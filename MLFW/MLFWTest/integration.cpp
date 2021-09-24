#include "pch.h"
#include "common.h"
#include <vector>
#include <operation/Conv.h>
#include "test_data/conv1d_sgd_integration.h"

TEST(Integration, Conv1d_SGD) {
	Conv1d conv(1, 1, 3);
	conv.weight->setData(conv1d_sgd_weight_start);
	conv.bias->setData(conv1d_sgd_bias_start);

	std::vector<int> in_shape = { 13, 1, 100 };
	std::vector<int> out_shape = { 13, 1, 98 };
	Tensor input(in_shape, conv1d_sgd_input);
	Tensor output(out_shape);
	conv.run(&output, &input);
	Tensor::sync();
	vector<float> output_data = output.getData();
	Tensor::sync();
	compare_arrays(conv1d_sgd_target, &output_data[0], output_data.size());
	Tensor::reset();
}
