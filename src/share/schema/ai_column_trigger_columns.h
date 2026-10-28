/* Copyright (c) 2026 OceanBase. Licensed under the Apache License, Version 2.0. */
#pragma once
#include "share/schema/ob_column_schema.h"
#include "share/schema/ob_table_schema.h"
namespace oceanbase
{
namespace share
{
namespace schema
{
inline bool ai_column_trigger_visible(const ObTableSchema &table, const ObColumnSchemaV2 &column)
{
  if (!column.is_hidden() || column.is_ai_internal_column())
    return true;
  if (column.get_column_id() != common::OB_HIDDEN_PK_INCREMENT_COLUMN_ID)
    return false;
  for (auto it = table.column_begin(); it != table.column_end(); ++it)
    if ((*it)->is_ai_column())
      return true;
  return false;
}
} // namespace schema
} // namespace share
} // namespace oceanbase
