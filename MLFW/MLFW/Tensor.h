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
	vector<float> data;
	vector<int> shape;
	int dims;
	int size;
	CUDATensor* cuda_data;
	CUDATensor* cuda_grad;
	CUDATensor* cuda_sens;
	CUDATensor createCudaTensor();
	void loadData(CUDATensor* src);
public:
	Tensor(vector<int> shape, float* data);
	vector<float> getData();
	vector<float> getGrad();
	vector<float> getSens();

	static void sync();
	static void reset();
};
