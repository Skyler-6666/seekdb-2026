/* Copyright (c) 2026 OceanBase. Licensed under the Apache License, Version 2.0. */
#pragma once
#include "share/schema/ob_table_schema.h"
#include "sql/parser/parse_node.h"

namespace oceanbase
{
namespace sql
{
namespace ai_column
{
int resolve_attribute(const ParseNode &attribute, share::schema::ObColumnSchemaV2 &column);
int finalize_schema(share::schema::ObTableSchema &schema,
                    const share::schema::ObTableSchema *original = nullptr);
} // namespace ai_column
} // namespace sql
} // namespace oceanbase
