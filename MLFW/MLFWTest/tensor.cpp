#include "pch.h"
#include "Tensor.h"

TEST(Tensor, Instantiation) {
	float data[] = { 3 };
	Tensor t(vector<int>({ 1 }), data);
	EXPECT_EQ(true, true);
}