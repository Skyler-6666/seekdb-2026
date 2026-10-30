/*
 * Copyright (c) 2025 OceanBase.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gtest/gtest.h>
#include "sql/engine/expr/ob_expr_ai/ob_ai_func_utils.h"

using namespace oceanbase::common;

namespace {
class AIPromptTest : public ::testing::Test
{
protected:
  int render(const char *json, ObString &result)
  {
    ObJsonObject *prompt = nullptr;
    ObString text(json);
    int ret = ObAIFuncJsonUtils::get_json_object_form_str(allocator_, text, prompt);
    if (OB_SUCCESS == ret) {
      ret = ObAIFuncPromptObjectUtils::replace_all_str_args_in_template(allocator_, prompt, result);
    }
    return ret;
  }
  ObArenaAllocator allocator_;
};

TEST_F(AIPromptTest, repeated_placeholder_exceeds_old_allocation)
{
  ObString result;
  ASSERT_EQ(OB_SUCCESS, render(R"({"template":"{0}{0}{0}{0}","args":["abcdefghijklmnopqrstuvwxyz"]})", result));
  EXPECT_TRUE(result == ObString("abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz"));
}

TEST_F(AIPromptTest, json_arrays_objects_and_nested_prompts)
{
  ObString result;
  ASSERT_EQ(OB_SUCCESS, render(R"({"template":"{0}|{1}|{2}","args":[[1,true],{"x":"y"},{"template":"inner {0}","args":["value"]}]})", result));
  EXPECT_TRUE(result == ObString("[1, true]|{\"x\": \"y\"}|inner value"));
}

TEST_F(AIPromptTest, unicode_escapes_and_literal_replacement)
{
  ObString result;
  ASSERT_EQ(OB_SUCCESS, render(R"({"template":"\u4f60\u597d {0}\n{0}","args":["{1}\\\"\u4e16\u754c"]})", result));
  EXPECT_TRUE(result == ObString("你好 {1}\\\"世界\n{1}\\\"世界"));
}

TEST_F(AIPromptTest, invalid_index_and_integer_overflow)
{
  ObString result;
  EXPECT_NE(OB_SUCCESS, render(R"({"template":"{1}","args":["x"]})", result));
  EXPECT_NE(OB_SUCCESS, render(R"({"template":"{-1}","args":["x"]})", result));
  EXPECT_NE(OB_SUCCESS, render(R"({"template":"{184467440737095516160}","args":["x"]})", result));
  EXPECT_NE(OB_SUCCESS, render(R"({"template":[],"args":["x"]})", result));
}

TEST_F(AIPromptTest, long_template_and_parameter)
{
  ObJsonArray *args = nullptr;
  ObJsonString *arg = nullptr;
  ObJsonObject *prompt = nullptr;
  char *data = static_cast<char *>(allocator_.alloc(100000));
  ASSERT_NE(nullptr, data);
  MEMSET(data, 'x', 100000);
  ObString value(100000, data);
  ObString tpl("{0}{0}{0}");
  ASSERT_EQ(OB_SUCCESS, ObAIFuncJsonUtils::get_json_array(allocator_, args));
  ASSERT_EQ(OB_SUCCESS, ObAIFuncJsonUtils::get_json_string(allocator_, value, arg));
  ASSERT_EQ(OB_SUCCESS, args->append(arg));
  ASSERT_EQ(OB_SUCCESS, ObAIFuncPromptObjectUtils::construct_prompt_object(allocator_, tpl, args, prompt));
  ObString result;
  ASSERT_EQ(OB_SUCCESS, ObAIFuncPromptObjectUtils::replace_all_str_args_in_template(allocator_, prompt, result));
  EXPECT_EQ(300000, result.length());
}
} // namespace
