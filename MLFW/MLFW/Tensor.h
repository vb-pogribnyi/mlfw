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
	void downloadData(CUDATensor* src);
	void uploadData(float* data, CUDATensor* dst);
	void reshapeCUDA(vector<int> new_shape, CUDATensor* dst);
public:
	Tensor(vector<int> shape, float* data = 0);
	void setData(float* data);
	void setGrad(float* data);
	vector<float> getData();
	vector<float> getGrad();
	vector<float> getSens();
	vector<int> getShape();
	int getSize();
	CUDATensor* getCudaData();
	CUDATensor* getCudaGrad();
	CUDATensor* getCudaSens();
	void clear(bool only_grad=false);
	Tensor* squeeze(int axis = -1);
	Tensor* unsqueeze(int axis = -1);
	void reshape(vector<int> new_shape);

	static void sync();
	static void reset();
};
