#include "pch.h"
#include "common.h"

void compare_arrays(float* arr1, float* arr2, int n) {
	for (int i = 0; i < n; i++) {
		float diff = abs(arr1[i] - arr2[i]);
		EXPECT_LE(diff, 2e-5);
	}
}

void compare_arrays(int* arr1, int* arr2, int n) {
	for (int i = 0; i < n; i++) {
		float diff = abs(arr1[i] - arr2[i]);
		EXPECT_LE(diff, 2e-5);
	}
}
