/* Copyright (c) 2026 OceanBase. Licensed under the Apache License, Version 2.0. */
#include "share/ai_service/ai_column_definition.h"
#include "share/schema/ob_column_schema.h"
#include <gtest/gtest.h>
using namespace oceanbase::common;
using namespace oceanbase::share::schema;
namespace ac = oceanbase::share::ai_column;
TEST(AIColumnSchema, CopySerializeAndRemap)
{
  ac::Definition definition;
  definition.model = "model";
  definition.expression = "AI_COMPLETE('model',body)";
  definition.input_expression = "body";
  definition.config_expression = "NULL";
  definition.version = "definition-1";
  definition.state_column_id = 17;
  definition.request_column_id = 18;
  definition.result_spec_column_id = 19;
  definition.state_column_name = "__ai_s_1";
  definition.request_column_name = "__ai_r_1";
  definition.result_spec_column_name = "__ai_v_1";
  definition.dependencies = {{"body", 20}};
  std::string encoded = ac::encode(definition);
  ObColumnSchemaV2 source;
  source.set_table_id(100001);
  source.set_column_id(16);
  source.set_prev_column_id(UINT64_MAX);
  source.set_next_column_id(UINT64_MAX);
  source.set_data_type(ObJsonType);
  source.add_column_flag(AI_COLUMN_FLAG);
  ASSERT_EQ(OB_SUCCESS, source.set_column_name("result"));
  ASSERT_EQ(OB_SUCCESS, source.set_ai_column_info(ObString(encoded.size(), encoded.data())));
  ObColumnSchemaV2 copy;
  ASSERT_EQ(OB_SUCCESS, copy.assign(source));
  ASSERT_TRUE(copy.is_ai_column());
  ASSERT_FALSE(copy.is_hidden());
  EXPECT_EQ(source.get_ai_column_info_str(), copy.get_ai_column_info_str());
  std::vector<char> buffer(source.get_serialize_size());
  int64_t written = 0, read = 0;
  ASSERT_EQ(OB_SUCCESS, source.serialize(buffer.data(), buffer.size(), written));
  ObColumnSchemaV2 restored;
  ASSERT_EQ(OB_SUCCESS, restored.deserialize(buffer.data(), written, read));
  EXPECT_EQ(written, read);
  EXPECT_EQ(source.get_ai_column_info_str(), restored.get_ai_column_info_str());
  hash::ObHashMap<uint64_t, uint64_t> ids;
  ASSERT_EQ(OB_SUCCESS, ids.create(8, "AIColumnTest", "AIColumnTest"));
  for (uint64_t id = 16; id <= 20; ++id)
    ASSERT_EQ(OB_SUCCESS, ids.set_refactored(id, id + 100));
  ASSERT_EQ(OB_SUCCESS, copy.convert_column_id(ids));
  EXPECT_EQ(116, copy.get_column_id());
  auto info = copy.get_ai_column_info_str();
  auto mapped = ac::decode(std::string(info.ptr(), info.length()));
  EXPECT_EQ(117, mapped.state_column_id);
  EXPECT_EQ(118, mapped.request_column_id);
  EXPECT_EQ(119, mapped.result_spec_column_id);
  EXPECT_EQ(120, mapped.dependencies[0].id);
  ObColumnSchemaV2 hidden;
  hidden.add_column_flag(AI_INTERNAL_COLUMN_FLAG);
  hidden.set_is_hidden(true);
  EXPECT_TRUE(hidden.is_ai_internal_column());
  EXPECT_TRUE(hidden.is_hidden());
  EXPECT_FALSE(hidden.is_ai_column());
}
