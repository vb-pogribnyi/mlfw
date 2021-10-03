#include "pch.h"
#include "common.h"
#include <operation/Conv.h>

TEST(Conv2d, Instantiation) {
	Conv2d conv(1, 1, 1);
	EXPECT_EQ(true, true);
}