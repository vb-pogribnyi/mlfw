#include "pch.h"
#include "common.h"
#include <operation/Conv.h>

TEST(Conv, Instantiation) {
	Conv1d conv(1, 1, 1);
	EXPECT_EQ(true, true);
}

//TEST(Conv, DimensionMismatch) {
//	Conv1d conv(1, 1, 3);
//
//	Tensor input(vector<int>({ 1, 1, 1 }));
//	Tensor output(vector<int>({ 1, 1, 1 }));
//	EXPECT_THROW(conv.run(&input, &output), TensorShapeError);
//}

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
	conv.run(&input, &output);
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
		{ 4.8, 5.7, 6.6, 10.2, 12.3, 14.4 }, { 1, 2, 3 } })//,
	//ConvRunTestCase({ { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115 }, { 2, 3, 5 },
	//	{ 0.5, 1, 0.5 }, { 1 }, {3, 2, 3},
	//	{ 4 }, { 2, 2, 2 } })
));
