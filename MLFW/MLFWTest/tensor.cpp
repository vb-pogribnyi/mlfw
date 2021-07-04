#include "pch.h"
#include "Tensor.h"

void compare_arrays(float* arr1, float* arr2, int n) {
	for (int i = 0; i < n; i++) {
		float diff = abs(arr1[i] - arr2[i]);
		EXPECT_LE(diff, 1e-5);
	}
}

TEST(Tensor, Instantiation) {
	float data[] = { 3 };
	Tensor t(vector<int>({ 1 }), data);
	EXPECT_EQ(true, true);
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