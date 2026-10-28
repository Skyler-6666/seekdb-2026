/* Copyright (c) 2026 OceanBase. Licensed under the Apache License, Version 2.0. */
#pragma once
#include <cstdint>
namespace oceanbase
{
namespace sql
{
class ObExecContext;
namespace ai_column
{
int install(uint64_t table_id);
int start_worker();
void stop_worker();
void wait_worker();
} // namespace ai_column
} // namespace sql
} // namespace oceanbase
