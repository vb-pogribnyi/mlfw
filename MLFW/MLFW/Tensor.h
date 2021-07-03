#pragma once

#include <vector>

using namespace std;

struct CUDATensor
{
	float* data;
	int* shape;
	int dims;
	int size;
};

class Tensor
{
private:
	float* data;
	vector<int> shape;
	int dims;
	int size;
	CUDATensor* cuda_data;
	CUDATensor* cuda_grad;
	CUDATensor* cuda_sens;
public:
	Tensor(vector<int> shape, float* data);
};
