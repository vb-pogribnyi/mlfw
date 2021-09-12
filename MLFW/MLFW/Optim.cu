#include "Optim.h"

Optimizer::Optimizer(list<Tensor*> tensors, float lr) : tensors(tensors), lr(lr) {}
