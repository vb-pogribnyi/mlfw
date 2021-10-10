#include "pch.h"
#include "common.h"
#include <operation/Conv.h>

TEST(Conv1d, Instantiation) {
	Conv1d conv(1, 1, 1);
	EXPECT_EQ(true, true);
}

TEST(Conv1d, DimensionMismatch) {
	Conv1d conv(1, 1, 3);

	Tensor input(vector<int>({ 1, 1, 1 }));
	Tensor output(vector<int>({ 1, 1, 1 }));
	EXPECT_THROW(conv.run(&output, &input), TensorShapeError);
}

struct Conv1dRunTestCase {
	vector<float> input;
	vector<int> in_shape;
	vector<float> kernel_w;
	vector<float> kernel_b;
	vector<int> kernel_shape;
	vector<float> output;
	vector<int> out_shape;
};

class Conv1dRunTest : public ::testing::TestWithParam<Conv1dRunTestCase> {};

TEST_P(Conv1dRunTest, Conv1dRunTest) {
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

INSTANTIATE_TEST_CASE_P(Conv1d, Conv1dRunTest, ::testing::Values(
	//Conv1dRunTestCase({ { 3 }, { 1, 1, 1 }, { 1 }, { 1 }, {1, 1, 1}, { 4 }, { 1, 1, 1 } }),
	//Conv1dRunTestCase({ { 2 }, { 1, 1, 1 }, { 1, 0.5 }, { 1, 1 }, {1, 2, 1}, { 3, 2 }, { 1, 2, 1 } }),
	//Conv1dRunTestCase({ { 2 }, { 1, 1, 1 }, { 1, 0.5 }, { 1, 2 }, {1, 2, 1}, { 3, 3 }, { 1, 2, 1 } }),
	Conv1dRunTestCase({ { 1, 1, 1 }, { 1, 1, 3 },
		{ 1, 1, 1 }, { 1 }, {1, 1, 3},
		{ 4 }, { 1, 1, 1 } }),
	Conv1dRunTestCase({ { 2, 1, 2 }, { 1, 1, 3 },
		{ 0.5, 1, 0.5 }, { 1 }, {1, 1, 3},
		{ 4 }, { 1, 1, 1 } }),
	Conv1dRunTestCase({ { 1, 2, 3, 6, 7, 8 }, { 1, 2, 3 },
		{ 0.1, 0.1, 0.1, 0.2, 0.2, 0.2 }, { 0 }, {2, 1, 3},
		{ 4.8 }, { 1, 1, 1 } }),
	Conv1dRunTestCase({ { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }, { 1, 2, 5 },
		{ 0.1, 0.1, 0.1, 0.2, 0.2, 0.2 }, { 0 }, {2, 1, 3},
		{ 4.8, 5.7, 6.6 }, { 1, 1, 3 } }),
	Conv1dRunTestCase({ { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }, { 1, 2, 5 },
		{ 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.4, 0.4, 0.4 }, { 0, 0 }, {2, 2, 3},
		{ 4.8, 5.7, 6.6, 10.2, 12.3, 14.4 }, { 1, 2, 3 } }),
	Conv1dRunTestCase({ { 1, 2, 3, 4, 5, 6 }, { 2, 1, 3 },
		{ 0.1, 0.2, 0.3 }, { 1 }, {1, 1, 3},
		{ 2.4, 4.2 }, { 2, 1, 1 } }),
	Conv1dRunTestCase({ { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }, { 2, 2, 3 },
		{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 }, { 1 }, {2, 1, 3},
		{ 10.1, 22.7 }, { 2, 1, 1 } }),
	Conv1dRunTestCase({ { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }, { 2, 1, 3 },
		{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 }, { 1, 1 }, {1, 2, 3},
		{ 2.4, 4.2, 4.2, 8.7 }, { 2, 2, 1 } }),
	Conv1dRunTestCase({ { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115 }, { 2, 3, 5 },
		{ -0.1, 0.2, -0.1, -0.2, 0.4, -0.2, -0.3, 0.6, -0.3, -0.4, 0.8, -0.4, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 }, { 1, 1 }, {3, 2, 3},
		{ 1., 1., 1., 23.6, 25.7, 27.8, 1., 1., 1., 233.6, 235.7, 237.8 }, { 2, 2, 3 } })
));

struct Conv1dGradTestCase {
	vector<float> input;
	vector<int> in_shape;
	vector<int> kernel_shape;
	vector<int> out_shape;
	vector<float> grad_input;
	vector<float> grad_w;
	vector<float> grad_b;
};

class Conv1dGradTest : public ::testing::TestWithParam<Conv1dGradTestCase> {};

TEST_P(Conv1dGradTest, Conv1dGradTest) {
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

INSTANTIATE_TEST_CASE_P(Conv1d, Conv1dGradTest, ::testing::Values(
	Conv1dGradTestCase({ { 1 }, { 1, 1, 1 }, {1, 1, 1}, {1, 1, 1}, { 1 }, {1}, {1} }),
	Conv1dGradTestCase({ { 1, 0.6 }, { 2, 1, 1 }, {1, 1, 1}, {2, 1, 1}, { 1, 1 }, {0.8}, {1} }),
	Conv1dGradTestCase({ { 1, 0.6 }, { 1, 2, 1 }, {2, 1, 1}, {1, 1, 1}, { 1.12 }, {1.12, 0.672}, {1.12} }),
	Conv1dGradTestCase({ { 0.6 }, { 1, 1, 1 }, {1, 2, 1}, {1, 2, 1}, { 0.84, 1.72 }, {0.252, 0.516}, {0.42, 0.86} }),
	Conv1dGradTestCase({ { 0.6, 0.2 }, { 1, 2, 1 }, {2, 2, 1},
		{1, 2, 1}, { 0.88, 1.52 }, {0.264, 0.088, 0.456, 0.152}, {0.44, 0.76} }),
	Conv1dGradTestCase({ { 0.6, 0.2, 0.8 }, { 1, 1, 3 }, {1, 1, 3},
		{1, 1, 1}, { 1.36 }, {0.816, 0.272, 1.088}, {1.36} }),
	Conv1dGradTestCase({ { 0.6, 0.2, 0.8, 0.3, 0.5, 0.4 }, { 1, 2, 3 }, {2, 2, 3},
		{1, 2, 1}, { 2.76, 4.54 }, {
			0.828, 0.276, 1.104,
			0.414, 0.69, 0.552,
			1.362, 0.454, 1.816,
			0.681, 1.135, 0.908
		}, {1.38, 2.27} }),
	Conv1dGradTestCase({ {
		0.6, 0.2, 0.8, 0.3, 0.5, 0.4,
		0.2, 0.4, 0.8, 0.4, 0.2, 0.7 }, { 2, 2, 3 }, {2, 2, 3},
		{2, 2, 1}, { 2.76, 4.54, 3.2, 3.36 }, {
			0.5740, 0.4580, 1.1920,
			0.5270, 0.5050, 0.8360,
			0.8490, 0.5630, 1.5800,
			0.6765, 0.7355, 1.0420
		}, {1.4900, 1.9750} }),
	Conv1dGradTestCase({ {
		0.6, 0.5, 0.2, 0.1, 0.8, 0.3, 0.5, 0.4,
		0.2, 0.4, 0.8, 0.6, 0.4, 0.3, 0.2, 0.7 }, { 2, 2, 4 }, {2, 2, 3},
		{2, 2, 2}, { 3.06, 2.7, 4.36, 3.04, 2.48, 3.72, 2.74, 5.24 }, {
			0.6462500691413879, 0.7547500729560852, 0.6372500658035278,
			0.6707500219345093, 0.46950003504753113, 0.7137500643730164,
			0.8474999666213989, 1.0095000267028809, 0.8140000104904175,
			0.8834999799728394, 0.5872499942779541, 0.9514999985694885
		}, {1.4950, 1.9225} })
));

struct Conv1dSensTestCase {
	vector<float> input;
	vector<int> in_shape;
	vector<float> kernel_w;
	vector<float> kernel_b;
	vector<int> kernel_shape;
	vector<int> out_shape;
	vector<float> grad_input;
	vector<float> sens;
};

class Conv1dSensTest : public ::testing::TestWithParam<Conv1dSensTestCase> {};

TEST_P(Conv1dSensTest, Conv1dSensTest) {
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

INSTANTIATE_TEST_CASE_P(Conv1d, Conv1dSensTest, ::testing::Values(
	Conv1dSensTestCase({ { 1 }, { 1, 1, 1 }, {0.2}, {1}, {1, 1, 1}, {1, 1, 1}, { 0.64 }, {0.128} }),
	Conv1dSensTestCase({ { 1 }, { 1, 1, 1 }, {0.2}, {0.4}, {1, 1, 1}, {1, 1, 1}, { -0.56 }, {-0.112} }),
	Conv1dSensTestCase({ { 1, 1 }, { 2, 1, 1 }, {0.2}, {1}, {1, 1, 1}, {2, 1, 1},
		{ 0.64, 1.08 }, {0.064, 0.108} }),
	Conv1dSensTestCase({ { 1 }, { 1, 1, 1 }, {0.4}, {1}, {1, 1, 1}, {1, 1, 1}, { 0.68 }, {0.272} }),
	Conv1dSensTestCase({
		{ 1, 1, 1 }, { 1, 1, 3 },
		{0.4, 0.6, 0.3}, {1}, {1, 1, 3},
		{1, 1, 1}, { 1.4 }, {0.56, 0.84, 0.42} }),
	Conv1dSensTestCase({
		{ 1, 1, 1, 1 }, { 1, 1, 4 },
		{0.4, 0.6, 0.3}, {1}, {1, 1, 3},
		{1, 1, 2}, { 1.4, 1.42 }, {0.28, 0.704, 0.636, 0.213} }),
	Conv1dSensTestCase({
		{ 1, 1, 1, 1, 1, 1, 1, 1 }, { 1, 2, 4 },
		{0.4, 0.6, 0.3, 0.1, 0.4, 0.2}, {1}, {2, 1, 3},
		{1, 1, 2}, { 2.32, 2.46 }, {0.464, 1.188, 1.086, 0.369, 0.116, 0.587, 0.724, 0.246} }),
	Conv1dSensTestCase({
		{ 1, 1, 1, 1, 1, 1, 1, 1 }, { 1, 2, 4 },
		{0.4, 0.6, 0.3, 0.1, 0.4, 0.2}, {0.6}, {2, 1, 3},
		{1, 1, 2}, { 1.52, 1.66 }, {
			0.3040000796318054,
			0.7880001664161682,
			0.7260001301765442,
			0.24900002777576447,
			0.07600001990795135,
			0.38700008392333984,
			0.4840000569820404,
			0.16600000858306885,
		} }),
	Conv1dSensTestCase({
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }, { 2, 2, 4 },
		{0.4, 0.6, 0.3, 0.1, 0.4, 0.2, 0.5, 0.6, 0.4, 0.1, 0.7, 0.3}, {1}, {2, 2, 3},
		{2, 2, 2}, { 2.32, 2.46, 2.36, 2.76, 3.6, 3.64, 2.46, 3.98 }, {
			0.26350003480911255,
			0.6465001106262207,
			0.5965000987052917,
			0.2302500307559967,
			0.058500006794929504,
			0.3877500295639038,
			0.5110000371932983,
			0.16500002145767212,
			0.33375000953674316,
			0.8852500319480896,
			0.8295000195503235,
			0.33550000190734863,
			0.07575000077486038,
			0.49049997329711914,
			0.7124999761581421,
			0.24025000631809235
		} })
));
