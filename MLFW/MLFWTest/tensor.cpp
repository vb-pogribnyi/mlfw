#include "pch.h"
#include "Tensor.h"
#include "common.h"

TEST(Tensor, Instantiation) {
	float data[] = { 3 };
	Tensor t(vector<int>({ 1 }), data);
	EXPECT_EQ(true, true);
}

TEST(Tensor, Instantiation_2) {
	float data[] = { 3, 3, 3 };
	vector<int> shape = { 1, 2 };
	Tensor t(shape, data);
	compare_arrays(&shape[0], &(t.getShape()[0]), shape.size());
	EXPECT_EQ(t.getSize(), 2);
}

TEST(Tensor, getData_2) {
	float expected_data[] = { 0 };
	Tensor t(vector<int>({ 1 }));
	compare_arrays(expected_data, &(t.getData()[0]), 1);
	Tensor::sync();
	Tensor::reset();
}

TEST(Tensor, getData) {
	float data[] = { 3 };
	float expected_grad[] = { 0 };
	Tensor t(vector<int>({ 1 }), data);
	compare_arrays(data, &(t.getData()[0]), 1);
	compare_arrays(expected_grad, &(t.getGrad()[0]), 1);
	compare_arrays(expected_grad, &(t.getSens()[0]), 1);
	Tensor::sync();
	Tensor::reset();
}

TEST(Tensor, setData) {
	float data[] = { 3 };
	float new_data[] = { 5 };
	float expected_grad[] = { 0 };
	Tensor t(vector<int>({ 1 }), data);
	t.setData(new_data);
	compare_arrays(new_data, &(t.getData()[0]), 1);
	Tensor::sync();
	Tensor::reset();
}