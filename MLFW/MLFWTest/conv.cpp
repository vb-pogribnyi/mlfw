#include "pch.h"
#include <operation/Conv.h>

TEST(Conv, Instantiation) {
	Conv1d conv(1, 1, 1);
	EXPECT_EQ(true, true);
}
