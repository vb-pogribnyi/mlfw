#include "pch.h"
#include "common.h"
#include <vector>
#include <operation/Conv.h>
#include <optimizer/SGD.h>
#include <operation/MSELoss.h>
#include "test_data/conv1d_sgd_integration.h"

#define TRAIN_ITERATION(weight_target, bias_target, loss_target) \
		conv.run(&output, &input); \
		mse.run(&mse_output, &output, &target); \
		mse_output_data = mse_output.getData(); \
		compare_arrays(loss_target, &mse_output_data[0], mse_output_data.size()); \
		mse.propagate(); \
		conv.propagate(); \
		opt.step(); \
		weight0 = conv.weight->getData(); \
		bias0 = conv.bias->getData(); \
		compare_arrays(weight_target, &weight0[0], weight0.size()); \
		compare_arrays(bias_target, &bias0[0], bias0.size()); \

#define TRAIN_ITERATION_TEST(idx) \
	conv1d_sgd_bias_idx

TEST(Integration, Conv1d_SGD) {
	Conv1d conv(1, 1, 3);
	SGD opt(list<Tensor*>({ conv.weight, conv.bias }), 1e-2);
	MSELoss mse;
	conv.weight->setData(conv1d_sgd_weight_start);
	conv.bias->setData(conv1d_sgd_bias_start);

	std::vector<int> in_shape = { 13, 1, 100 };
	std::vector<int> out_shape = { 13, 1, 98 };
	std::vector<int> loss_shape = { 1, 1, 1 };
	Tensor input(in_shape, conv1d_sgd_input);
	Tensor output(out_shape);
	Tensor target(out_shape, conv1d_sgd_target);
	Tensor mse_output(loss_shape);
	conv.run(&output, &input);
	Tensor::sync();
	mse.run(&mse_output, &output, &target);
	Tensor::sync();
	vector<float> output_data = output.getData();
	vector<float> mse_output_data = mse_output.getData();
	Tensor::sync();
	compare_arrays(conv1d_sgd_out, &output_data[0], output_data.size());
	compare_arrays(conv1d_sgd_loss0, &mse_output_data[0], mse_output_data.size());

	mse.propagate();
	conv.propagate();
	opt.step();
	vector<float> weight0 = conv.weight->getData();
	vector<float> bias0 = conv.bias->getData();
	compare_arrays(conv1d_sgd_weight0, &weight0[0], weight0.size());
	compare_arrays(conv1d_sgd_bias0, &bias0[0], bias0.size());

	TRAIN_ITERATION(conv1d_sgd_weight1, conv1d_sgd_bias1, conv1d_sgd_loss1);
	TRAIN_ITERATION(conv1d_sgd_weight2, conv1d_sgd_bias2, conv1d_sgd_loss2);
	TRAIN_ITERATION(conv1d_sgd_weight3, conv1d_sgd_bias3, conv1d_sgd_loss3);
	TRAIN_ITERATION(conv1d_sgd_weight4, conv1d_sgd_bias4, conv1d_sgd_loss4);
	TRAIN_ITERATION(conv1d_sgd_weight5, conv1d_sgd_bias5, conv1d_sgd_loss5);
	TRAIN_ITERATION(conv1d_sgd_weight6, conv1d_sgd_bias6, conv1d_sgd_loss6);
	TRAIN_ITERATION(conv1d_sgd_weight7, conv1d_sgd_bias7, conv1d_sgd_loss7);
	TRAIN_ITERATION(conv1d_sgd_weight8, conv1d_sgd_bias8, conv1d_sgd_loss8);
	TRAIN_ITERATION(conv1d_sgd_weight9, conv1d_sgd_bias9, conv1d_sgd_loss9);

	Tensor::sync();
	Tensor::reset();
}
