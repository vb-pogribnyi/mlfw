#include "Operation.h"

void Operation::record_flow(Tensor* output, Tensor* input1, Tensor* input2) {
	flow_input1 = input1;
	flow_input2 = input2;
	flow_output = output;
}
