#include "pch.h"
#include "common.h"
#include <operation/MSELoss.h>

TEST(MSELoss, Instantiation) {
	MSELoss mse;
	EXPECT_EQ(true, true);
}

struct MSELossRunTestCase {
	vector<float> input;
	vector<int> in_shape;
	vector<float> target;
	vector<int> target_shape;
	vector<float> output;
	vector<int> out_shape;
};

class MSELossTest : public testing::TestWithParam<MSELossRunTestCase> {};

TEST_P(MSELossTest, MSELossTest) {
	auto p = GetParam();
	MSELoss mse;

	Tensor input(p.in_shape, &p.input[0]);
	Tensor target(p.target_shape, &p.target[0]);
	Tensor output(p.out_shape);
	mse.run(&output , &input, &target);
	Tensor::sync();
	vector<float> output_data = output.getData();
	Tensor::sync();
	compare_arrays(&p.output[0], &output_data[0], p.output.size());
	Tensor::reset();
}

INSTANTIATE_TEST_CASE_P(MSELossTest, MSELossTest, testing::Values(
	MSELossRunTestCase({ {0}, {1, 1, 1},  {1}, {1, 1, 1}, {1}, {1, 1, 1} }),
	MSELossRunTestCase({ {1}, {1, 1, 1},  {0}, {1, 1, 1}, {1}, {1, 1, 1} }),
	MSELossRunTestCase({ {1}, {1, 1, 1},  {1}, {1, 1, 1}, {0}, {1, 1, 1} }),
	MSELossRunTestCase({ {-1, 4}, {1, 1, 2}, {0, 1}, {1, 1, 2}, {5}, {1, 1, 1} }),
	MSELossRunTestCase({ {-1, 4}, {1, 2, 1}, {0, 1}, {1, 2, 1}, {5}, {1, 1, 1} }),
	MSELossRunTestCase({ {-1, 4, 0, 3}, {2, 2, 1}, {0, 1, 2, 1}, {2, 2, 1}, {4.5}, {1, 1, 1} })
));


class MSELossBackTest : public testing::TestWithParam<MSELossRunTestCase> {};

TEST_P(MSELossBackTest, MSELossBackTest) {
	auto p = GetParam();
	MSELoss mse;

	Tensor input(p.in_shape, &p.input[0]);
	Tensor target(p.target_shape, &p.target[0]);
	Tensor output(p.out_shape);
	mse.run(&output, &input, &target);
	mse.propagate();
	Tensor::sync();
	vector<float> output_data = input.getGrad();
	Tensor::sync();
	compare_arrays(&p.output[0], &output_data[0], p.output.size());
	Tensor::reset();
}

INSTANTIATE_TEST_CASE_P(MSELossBackTest, MSELossBackTest, testing::Values(
	MSELossRunTestCase({ {1}, {1, 1, 1},  {1.2}, {1, 1, 1}, {-0.4}, {1, 1, 1} }),
	MSELossRunTestCase({ {0.8}, {1, 1, 1},  {1.2}, {1, 1, 1}, {-0.8}, {1, 1, 1} }),
	MSELossRunTestCase({ {0.8, 1.0}, {2, 1, 1},  {1.2, 1.2}, {2, 1, 1}, {-0.8, -0.4}, {2, 1, 1} }),
	MSELossRunTestCase({ {0.8, 1.0}, {1, 2, 1},  {1.2, 1.2}, {1, 2, 1}, {-0.4, -0.2}, {1, 2, 1} })
));
