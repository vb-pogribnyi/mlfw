#pragma once
#include <list>
#include "Tensor.h"

using namespace std;

class Optimizer
{
public:
	Optimizer(list<Tensor*> tensors, float lr);
	virtual void step() = 0;
protected:
	list<Tensor*> tensors;
	float lr;
};
