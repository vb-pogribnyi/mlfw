#include "pch.h"
#include "common.h"
#include <operation/Conv.h>

TEST(Conv2d, Instantiation) {
	Conv2d conv(1, 1, 1, 1);
	EXPECT_EQ(true, true);
}

struct Conv2dRunTestCase {
	vector<float> input;
	vector<int> in_shape;
	vector<float> kernel_w;
	vector<float> kernel_b;
	vector<int> kernel_shape;
	vector<float> output;
	vector<int> out_shape;
};

class Conv2dRunTest : public ::testing::TestWithParam<Conv2dRunTestCase> {};

TEST_P(Conv2dRunTest, Conv2dRunTest) {
	auto p = GetParam();
	vector<int> ks = p.kernel_shape;
	Conv2d conv(ks[0], ks[1], ks[2], ks[3]);
	conv.weight->setData(&p.kernel_w[0]);
	conv.bias->setData(&p.kernel_b[0]);

	Tensor input(p.in_shape, &p.input[0]);
	Tensor output(p.out_shape);
	conv.run(&output, &input);
	Tensor::sync();
	vector<float> output_data = output.getData();
	Tensor::sync();
	compare_arrays(&p.output[0], &output_data[0], p.output.size());
	Tensor::reset();
}

INSTANTIATE_TEST_CASE_P(Conv2d, Conv2dRunTest, ::testing::Values(
	Conv2dRunTestCase({ { 1 }, { 1, 1, 1, 1 }, { 1 }, { 1 }, {1, 1, 1, 1}, { 2 }, { 1, 1, 1, 1 } }),
	Conv2dRunTestCase({ { 1, 1, 1, 1, 1, 1, 1, 1, 1 }, { 1, 1, 3, 3 }, 
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1 }, { 1 }, {1, 1, 3, 3},
		{ 10 }, { 1, 1, 1, 1 } })
));