#pragma once

#include "../Optim.h"
#include <list>

using namespace std;

class SGD : public Optimizer
{
public:
	SGD(list<Tensor*> tensors, float lr);
	void step();
};
