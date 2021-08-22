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
	vector<float> kernel_w(ks[0] * ks[1] * ks[2], 1);
	vector<float> kernel_b(ks[1], 1);
	conv.weight->setData(&kernel_w[0]);
	conv.bias->setData(&kernel_b[0]);

	Tensor input(p.in_shape, &p.input[0]);
	Tensor output(p.out_shape);
	conv.run(&output, &input);
	output.setGrad(&p.grad_input[0]);

	conv.propagate();
	Tensor::sync();
	vector<float> real_grad_w = conv.weight->getGrad();
	compare_arrays(&p.grad_w[0], &real_grad_w[0], p.grad_w.size());
	vector<float> real_grad_b = conv.bias->getGrad();
	compare_arrays(&p.grad_b[0], &real_grad_b[0], p.grad_b.size());
	Tensor::reset();
}

INSTANTIATE_TEST_CASE_P(Conv, ConvGradTest, ::testing::Values(
	ConvGradTestCase({ { 1 }, { 1, 1, 1 }, {1, 1, 1}, {1, 1, 1}, { 1 }, {1}, {1} }),
	ConvGradTestCase({ { 1, 0.6 }, { 2, 1, 1 }, {1, 1, 1}, {2, 1, 1}, { 1, 1 }, {0.8}, {1} }),
	ConvGradTestCase({ { 1, 0.6 }, { 1, 2, 1 }, {2, 1, 1}, {1, 1, 1}, { 1.12 }, {1.12, 0.672}, {1.12} }),
	ConvGradTestCase({ { 0.6 }, { 1, 1, 1 }, {1, 2, 1}, {1, 2, 1}, { 0.84, 1.72 }, {0.252, 0.516}, {0.42, 0.86} }),
	ConvGradTestCase({ { 0.6, 0.2 }, { 1, 2, 1 }, {2, 2, 1},
		{1, 2, 1}, { 0.88, 1.52 }, {0.264, 0.456, 0.088, 0.152}, {0.44, 0.76} }),
	ConvGradTestCase({ { 0.6, 0.2, 0.8 }, { 1, 1, 3 }, {1, 1, 3},
		{1, 1, 1}, { 1.36 }, {0.816, 0.272, 1.088}, {1.36} }),
	ConvGradTestCase({ { 0.6, 0.2, 0.8, 0.3, 0.5, 0.4 }, { 1, 2, 3 }, {2, 2, 3},
		{1, 2, 1}, { 2.76, 4.54 }, {
			0.828, 0.276, 1.104,
			1.362, 0.454, 1.816,
			0.414, 0.69, 0.552,
			0.681, 1.135, 0.908
		}, {1.38, 2.27} }),
	ConvGradTestCase({ {
		0.6, 0.2, 0.8, 0.3, 0.5, 0.4,
		0.2, 0.4, 0.8, 0.4, 0.2, 0.7 }, { 2, 2, 3 }, {2, 2, 3},
		{2, 2, 1}, { 2.76, 4.54, 3.2, 3.36 }, {
			0.5740, 0.4580, 1.1920,
			0.8490, 0.5630, 1.5800,
			0.5270, 0.5050, 0.8360,
			0.6765, 0.7355, 1.0420
		}, {1.4900, 1.9750} }),
	ConvGradTestCase({ {
		0.6, 0.5, 0.2, 0.1, 0.8, 0.3, 0.5, 0.4,
		0.2, 0.4, 0.8, 0.6, 0.4, 0.3, 0.2, 0.7 }, { 2, 2, 4 }, {2, 2, 3},
		{2, 2, 2}, { 3.06, 2.7, 4.36, 3.04, 2.48, 3.72, 2.74, 5.24 }, {
			0.6462500691413879, 0.7547500729560852, 0.6372500658035278,
			0.8474999666213989, 1.0095000267028809, 0.8140000104904175,
			0.6707500219345093, 0.46950003504753113, 0.7137500643730164,
			0.8834999799728394, 0.5872499942779541, 0.9514999985694885
		}, {1.4950, 1.9225} })
));

struct ConvSensTestCase {
	vector<float> input;
	vector<int> in_shape;
	vector<float> kernel_w;
	vector<float> kernel_b;
	vector<int> kernel_shape;
	vector<int> out_shape;
	vector<float> grad_input;
	vector<float> sens;
};

class ConvSensTest : public ::testing::TestWithParam<ConvSensTestCase> {};

TEST_P(ConvSensTest, ConvSensTest) {
	auto p = GetParam();
	vector<int> ks = p.kernel_shape;
	Conv1d conv(ks[0], ks[1], ks[2]);
	conv.weight->setData(&p.kernel_w[0]);
	conv.bias->setData(&p.kernel_b[0]);

	Tensor input(p.in_shape, &p.input[0]);
	Tensor output(p.out_shape);
	conv.run(&output, &input);
	output.setGrad(&p.grad_input[0]);

	conv.propagate();
	Tensor::sync();
	vector<float> real_sens = input.getGrad();
	compare_arrays(&p.sens[0], &real_sens[0], p.sens.size());
	Tensor::reset();
}

//vector<float> input;
//vector<int> in_shape;
//vector<float> kernel_w;
//vector<float> kernel_b;
//vector<int> kernel_shape;
//vector<int> out_shape;
//vector<float> grad_input;
//vector<float> sens;


INSTANTIATE_TEST_CASE_P(Conv, ConvSensTest, ::testing::Values(
	ConvSensTestCase({ { 1 }, { 1, 1, 1 }, {0.1}, {1}, {1, 1, 1}, {1, 1, 1}, { 0.64 }, {0.128} })
));
