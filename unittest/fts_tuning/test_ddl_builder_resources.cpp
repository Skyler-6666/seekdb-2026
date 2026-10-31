/* Copyright (c) 2025 OceanBase.
 * Licensed under the Apache License, Version 2.0.
 */

#include <gtest/gtest.h>
#include "share/ob_ddl_task_executor.h"

namespace oceanbase
{
namespace share
{

TEST(DDLLocalBuilderResources, SmallMiniDeploymentsKeepOneWorker)
{
  const int64_t four_gib = 4LL << 30;
  // CPU and memory are independent limits; spare resources on one axis must
  // not enable concurrent builds when the other axis is constrained.
  EXPECT_EQ(1, ObDDLLocalBuilder::calc_thread_count(true, 0, 8LL << 30));
  EXPECT_EQ(1, ObDDLLocalBuilder::calc_thread_count(true, 1, 8LL << 30));
  EXPECT_EQ(1, ObDDLLocalBuilder::calc_thread_count(true, 3, 8LL << 30));
  EXPECT_EQ(1, ObDDLLocalBuilder::calc_thread_count(true, 4, four_gib - 1));
  EXPECT_EQ(1, ObDDLLocalBuilder::calc_thread_count(true, 32, 2LL << 30));
  EXPECT_EQ(1, ObDDLLocalBuilder::calc_thread_count(true, 4, 0));
}

TEST(DDLLocalBuilderResources, LargerMiniDeploymentsHaveBoundedConcurrency)
{
  EXPECT_EQ(2, ObDDLLocalBuilder::calc_thread_count(true, 4, 4LL << 30));
  EXPECT_EQ(2, ObDDLLocalBuilder::calc_thread_count(true, 4, 8LL << 30));
  EXPECT_EQ(2, ObDDLLocalBuilder::calc_thread_count(true, 64, 9LL << 30));
}

TEST(DDLLocalBuilderResources, NormalModePreservesExistingConcurrency)
{
  EXPECT_EQ(16, ObDDLLocalBuilder::calc_thread_count(false, 4, 16LL << 30));
  EXPECT_EQ(16, ObDDLLocalBuilder::calc_thread_count(false, 64, 64LL << 30));
  EXPECT_EQ(16, ObDDLLocalBuilder::calc_thread_count(false, 2, 16LL << 30));
}

} // namespace share
} // namespace oceanbase

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
