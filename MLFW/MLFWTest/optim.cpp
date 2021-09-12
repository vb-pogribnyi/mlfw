#include "pch.h"
#include "common.h"
#include <optimizer/SGD.h>

TEST(Optim, Instantiation) {
	SGD sgd({}, 1e-3);
	EXPECT_TRUE(true);
}

struct SGDStepTestCase {
	vector<float> weights_before;
	vector<float> weights_expected;
	vector<float> bias_before;
	vector<float> bias_expected;
	vector<float> grads_weight;
	vector<float> grads_bias;
	vector<int> kernel_shape;
	float lr;
};

class SGDStepTest : public testing::TestWithParam<SGDStepTestCase> {};

TEST_P(SGDStepTest, SGDStepTest) {
	auto param = GetParam();
	Tensor weight(param.kernel_shape, &param.weights_before[0]);
	Tensor bias({param.kernel_shape[1]}, &param.bias_before[0]);

	weight.setGrad(&param.grads_weight[0]);
	bias.setGrad(&param.grads_bias[0]);
	SGD opt(list<Tensor*>({ &weight, &bias }), param.lr);
	opt.step();
	Tensor::sync();

	vector<float> weight_after = weight.getData();
	vector<float> bias_after = bias.getData();
	compare_arrays(&param.weights_expected[0], &weight_after[0], param.weights_expected.size());
	compare_arrays(&param.bias_expected[0], &bias_after[0], param.bias_expected.size());
	Tensor::reset();
}

INSTANTIATE_TEST_CASE_P(OptimSGDStep, SGDStepTest, testing::Values(
	SGDStepTestCase({ {1}, {0.96}, {0.}, {-0.04}, {0.4}, {0.4}, {1, 1, 1}, 0.1 }),
	SGDStepTestCase({ {1., 0.2, 0.1, 0.3, 0.4, 1.}, {0.959, 0.1754, 0.0631, 0.126, 0.2956, 0.8434},
		{0., 0.6}, {-0.041, 0.426},
		{0.41, 0.246, 0.369, 1.74, 1.044, 1.566}, {0.41, 1.74}, {1, 2, 3}, 0.1 }),
	SGDStepTestCase({ {1., 0.2, 0.1, 0.3, 0.4, 1.}, {0.9584, 0.17504, 0.06256, 0.28752, 0.39168, 0.98336},
		{0.6}, {0.5584},
		{4.16, 2.496, 3.744, 1.248, 0.832, 1.664}, {4.16}, {2, 1, 3}, 0.01 })
));
