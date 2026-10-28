/* Copyright (c) 2026 OceanBase. Licensed under the Apache License, Version 2.0. */
#include "lib/allocator/ob_allocator.h"
#include "sql/parser/ob_parser.h"
#include <gtest/gtest.h>
using namespace oceanbase::common;
using namespace oceanbase::sql;
namespace
{
bool contains(const ParseNode *node, ObItemType type)
{
  if (node == nullptr)
    return false;
  if (node->type_ == type)
    return true;
  for (int i = 0; i < node->num_child_; ++i)
    if (contains(node->children_[i], type))
      return true;
  return false;
}
} // namespace
TEST(AIColumnParser, CreateAddModifyAndShow)
{
  for (const char *sql :
       {"CREATE TABLE docs(id BIGINT PRIMARY KEY, body TEXT, summary TEXT AI COLUMN "
        "(AI_COMPLETE('text_model',body)) WITH(sync_mode='async',max_retries=2))",
        "ALTER TABLE docs ADD emb VECTOR(4) AI COLUMN (AI_EMBED('embedding_model',body,4))",
        "ALTER TABLE docs MODIFY summary JSON AI COLUMN "
        "(AI_COMPLETE('text_model',concat('Classify: ',body)))",
        "SHOW AI COLUMNS FROM docs"})
  {
    ObArenaAllocator allocator;
    ObParser parser(allocator, 0);
    ParseResult result;
    ASSERT_EQ(OB_SUCCESS, parser.parse(ObString(sql), result)) << sql;
    EXPECT_TRUE(contains(result.result_tree_, T_AI_COLUMN) ||
                contains(result.result_tree_, T_SHOW_AI_COLUMNS));
  }
}
TEST(AIColumnParser, RetryPreservesBusinessStatusQuoting)
{
  for (const char *sql :
       {"ALTER TABLE docs RETRY AI COLUMN summary WHERE status='FAILED' AND id=1",
        "ALTER TABLE docs RETRY AI COLUMN summary WHERE `status`='active' AND id=1",
        "ALTER TABLE db.docs RETRY AI COLUMN summary"})
  {
    ObArenaAllocator allocator;
    ObParser parser(allocator, 0);
    ParseResult result;
    ASSERT_EQ(OB_SUCCESS, parser.parse(ObString(sql), result)) << sql;
    EXPECT_EQ(T_AI_COLUMN_RETRY, result.result_tree_->children_[0]->type_);
  }
}
TEST(AIColumnParser, OrdinarySqlKeepsRetryIdentifier)
{
  ObArenaAllocator allocator;
  ObParser parser(allocator, 0);
  ParseResult result;
  EXPECT_EQ(OB_SUCCESS,
            parser.parse(ObString("SELECT retry, status FROM ordinary_table WHERE id=1"), result));
}
