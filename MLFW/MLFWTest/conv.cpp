#include "pch.h"
#include "common.h"
#include <operation/Conv.h>

TEST(Conv, Instantiation) {
	Conv1d conv(1, 1, 1);
	EXPECT_EQ(true, true);
}

TEST(Conv, DimensionMismatch) {
	Conv1d conv(1, 1, 3);

	Tensor input(vector<int>({ 1, 1, 1 }));
	Tensor output(vector<int>({ 1, 1, 1 }));
	EXPECT_THROW(conv.run(&output, &input), TensorShapeError);
}

struct ConvRunTestCase {
	vector<float> input;
	vector<int> in_shape;
	vector<float> kernel_w;
	vector<float> kernel_b;
	vector<int> kernel_shape;
	vector<float> output;
	vector<int> out_shape;
};

class ConvRunTest : public ::testing::TestWithParam<ConvRunTestCase> {};

TEST_P(ConvRunTest, ConvRunTest) {
	auto p = GetParam();
	vector<int> ks = p.kernel_shape;
	Conv1d conv(ks[0], ks[1], ks[2]);
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

INSTANTIATE_TEST_CASE_P(Conv, ConvRunTest, ::testing::Values(
	ConvRunTestCase({ { 3 }, { 1, 1, 1 }, { 1 }, { 1 }, {1, 1, 1}, { 4 }, { 1, 1, 1 } }),
	ConvRunTestCase({ { 2 }, { 1, 1, 1 }, { 1, 0.5 }, { 1, 1 }, {1, 2, 1}, { 3, 2 }, { 1, 2, 1 } }),
	ConvRunTestCase({ { 2 }, { 1, 1, 1 }, { 1, 0.5 }, { 1, 2 }, {1, 2, 1}, { 3, 3 }, { 1, 2, 1 } }),
	ConvRunTestCase({ { 1, 1, 1 }, { 1, 1, 3 },
		{ 1, 1, 1 }, { 1 }, {1, 1, 3},
		{ 4 }, { 1, 1, 1 } }),
	ConvRunTestCase({ { 2, 1, 2 }, { 1, 1, 3 },
		{ 0.5, 1, 0.5 }, { 1 }, {1, 1, 3},
		{ 4 }, { 1, 1, 1 } }),
	ConvRunTestCase({ { 1, 2, 3, 6, 7, 8 }, { 1, 2, 3 },
		{ 0.1, 0.1, 0.1, 0.2, 0.2, 0.2 }, { 0 }, {2, 1, 3},
		{ 4.8 }, { 1, 1, 1 } }),
	ConvRunTestCase({ { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }, { 1, 2, 5 },
		{ 0.1, 0.1, 0.1, 0.2, 0.2, 0.2 }, { 0 }, {2, 1, 3},
		{ 4.8, 5.7, 6.6 }, { 1, 1, 3 } }),
	ConvRunTestCase({ { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }, { 1, 2, 5 },
		{ 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.4, 0.4, 0.4 }, { 0, 0 }, {2, 2, 3},
		{ 4.8, 5.7, 6.6, 10.2, 12.3, 14.4 }, { 1, 2, 3 } }),
	ConvRunTestCase({ { 1, 2, 3, 4, 5, 6 }, { 2, 1, 3 },
		{ 0.1, 0.2, 0.3 }, { 1 }, {1, 1, 3},
		{ 2.4, 4.2 }, { 2, 1, 1 } }),
	ConvRunTestCase({ { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }, { 2, 2, 3 },
		{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 }, { 1 }, {2, 1, 3},
		{ 10.1, 22.7 }, { 2, 1, 1 } }),
	ConvRunTestCase({ { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }, { 2, 1, 3 },
		{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 }, { 1, 1 }, {1, 2, 3},
		{ 2.4, 4.2, 4.2, 8.7 }, { 2, 2, 1 } }),
	ConvRunTestCase({ { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115 }, { 2, 3, 5 },
		{ -0.1, 0.2, -0.1, -0.2, 0.4, -0.2, -0.3, 0.6, -0.3, -0.4, 0.8, -0.4, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 }, { 1, 1 }, {3, 2, 3},
		{ 1., 1., 1., 23.6, 25.7, 27.8, 1., 1., 1., 233.6, 235.7, 237.8 }, { 2, 2, 3 } })
));

struct ConvGradTestCase {
	vector<float> input;
	vector<int> in_shape;
	vector<float> kernel_w;
	vector<float> kernel_b;
	vector<int> kernel_shape;
	vector<int> out_shape;
	vector<float> grad_input;
	vector<float> grad_w;
	vector<float> grad_b;
};

class ConvGradTest : public ::testing::TestWithParam<ConvGradTestCase> {};

TEST_P(ConvGradTest, ConvGradTest) {
	auto p = GetParam();
	vector<int> ks = p.kernel_shape;
	Conv1d conv(ks[0], ks[1], ks[2]);
	conv.weight->setData(&p.kernel_w[0]);
	conv.bias->setData(&p.kernel_b[0]);

	Tensor input(p.in_shape, &p.input[0]);
	Tensor output(p.out_shape);
	conv.run(&output, &input);
	output.setGrad(&p.grad_input[0]);

	auto result = output.getGrad();

	conv.propagate();
	Tensor::sync();
	vector<float> real_grad_w = conv.weight->getGrad();
	compare_arrays(&p.grad_w[0], &real_grad_w[0], p.grad_w.size());
	vector<float> real_grad_b = conv.bias->getGrad();
	compare_arrays(&p.grad_b[0], &real_grad_b[0], p.grad_b.size());
	Tensor::reset();
}

INSTANTIATE_TEST_CASE_P(Conv, ConvGradTest, ::testing::Values(
	//ConvGradTestCase({ { 1 }, { 1, 1, 1 }, { 1 }, { 1 }, {1, 1, 1}, {1, 1, 1}, { 1 }, {1}, {1} }),
	ConvGradTestCase({ { 1, 0.6 }, { 2, 1, 1 }, { 1 }, { 1 }, {1, 1, 1}, {2, 1, 1}, { 1, 1 }, {0.8}, {1} })
));
