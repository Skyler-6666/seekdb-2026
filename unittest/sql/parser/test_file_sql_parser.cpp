/* Copyright (c) 2026 OceanBase. Licensed under the Apache License, Version 2.0. */
#include <gtest/gtest.h>
#include "sql/parser/ob_parser.h"
#include "sql/parser/parse_node.h"
#include "lib/allocator/ob_allocator.h"

using namespace oceanbase::common;
using namespace oceanbase::sql;

TEST(FileSqlParser, table_functions_and_named_options)
{
  for (const char *sql : {
      "SELECT * FROM read_csv('/tmp/input.csv',header=true,sample_size=-1) AS data",
      "SELECT f.id FROM read_json('/tmp/input.jsonl') f",
      "SELECT * FROM read_ndjson('/tmp/input.jsonl',format='newline_delimited')",
      "SELECT count(*) FROM read_parquet('/tmp/input.parquet')",
      "SELECT file FROM glob('/tmp/*.csv') ORDER BY file"}) {
    ObArenaAllocator allocator;
    ObParser parser(allocator, 0);
    ParseResult parsed;
    ASSERT_EQ(OB_SUCCESS, parser.parse(ObString(sql), parsed)) << sql;
    ASSERT_NE(nullptr, parsed.result_tree_);
    EXPECT_EQ(T_SELECT, parsed.result_tree_->children_[0]->type_);
  }
}

TEST(FileSqlParser, describe_and_copy)
{
  for (const char *sql : {
      "DESCRIBE SELECT * FROM read_csv('/tmp/input.csv')",
      "COPY (SELECT id FROM read_csv('/tmp/input.csv') ORDER BY id) TO '/tmp/out.csv' (FORMAT CSV, HEADER, NULLSTR 'NULL')",
      "COPY local_table TO '/tmp/out.csv' (HEADER FALSE, DELIMITER ';', QUOTE '\"', ESCAPE '\"')",
      "SELECT 1", "EXPLAIN SELECT 1"}) {
    ObArenaAllocator allocator;
    ObParser parser(allocator, 0);
    ParseResult parsed;
    ASSERT_EQ(OB_SUCCESS, parser.parse(ObString(sql), parsed)) << sql;
  }
}

TEST(FileSqlParser, rejects_positional_options)
{
  ObArenaAllocator allocator;
  ObParser parser(allocator, 0);
  ParseResult parsed;
  EXPECT_NE(OB_SUCCESS, parser.parse(ObString("SELECT * FROM read_csv('/tmp/input.csv','xml')"), parsed));
}
