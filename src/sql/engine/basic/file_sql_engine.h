/* Copyright (c) 2026 OceanBase. Licensed under the Apache License, Version 2.0. */
#pragma once
#include "file_scan_utils.h"
#include "common/object/ob_obj_type.h"
#include "sql/parser/parse_node.h"
#include "sql/session/ob_sql_session_info.h"

namespace oceanbase { namespace sql { namespace file_sql {
int access_root(ObSQLSessionInfo &session, common::ObIAllocator &allocator, common::ObString &root);
Plan source_from_node(const ParseNode &node);
common::ObDataType engine_type(const Column &column);
}}}
