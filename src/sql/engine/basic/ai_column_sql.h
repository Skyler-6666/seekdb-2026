/* Copyright (c) 2026 OceanBase. Licensed under the Apache License, Version 2.0. */
#pragma once
#include "lib/container/ob_iarray.h"
#include "query/parser/parse_node.h"
namespace oceanbase
{
namespace sql
{
class ObExecContext;
class ObSQLSessionInfo;
struct ObResolverParams;
namespace ai_column
{
extern thread_local bool resolving_retry;
int rewrite(ObResolverParams &params, const ParseNode &source, const ParseNode *&target);
int check_write(ObSQLSessionInfo &session, uint64_t flags);
int strict_read(ObExecContext &context, const common::ObIArray<uint64_t> &references);
} // namespace ai_column
} // namespace sql
} // namespace oceanbase
