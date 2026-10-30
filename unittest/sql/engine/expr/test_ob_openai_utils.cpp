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
#include <limits>
#include "sql/engine/expr/ob_expr_ai/ob_ai_func_utils.h"

using namespace oceanbase::common;

namespace {
class AIProviderTest : public ::testing::Test
{
protected:
  ObJsonObject *object(const char *text)
  {
    ObString input(text);
    ObJsonObject *obj = nullptr;
    EXPECT_EQ(OB_SUCCESS, ObAIFuncJsonUtils::get_json_object_form_str(allocator_, input, obj));
    return obj;
  }
  ObArenaAllocator allocator_;
};

TEST_F(AIProviderTest, embedding_restores_indexes)
{
  ObOpenAIUtils::ObOpenAIEmbed provider;
  ObIJsonBase *result = nullptr;
  ASSERT_EQ(OB_SUCCESS, provider.parse_output(allocator_, object(R"({"data":[{"index":1,"embedding":[3,4]},{"index":0,"embedding":[1,2]}]})"), result));
  auto *array = static_cast<ObJsonArray *>(result);
  EXPECT_EQ(1, static_cast<ObJsonArray *>(array->get_value(0))->get_value(0)->get_int());
  EXPECT_EQ(3, static_cast<ObJsonArray *>(array->get_value(1))->get_value(0)->get_int());
  EXPECT_NE(OB_SUCCESS, ObAIFuncUtils::validate_embeddings(allocator_, result, 2, 3));
  EXPECT_EQ(OB_SUCCESS, ObAIFuncUtils::validate_embeddings(allocator_, result, 2, 2));
}

TEST_F(AIProviderTest, malformed_embedding_responses)
{
  const char *cases[] = {
    R"({"data":{}})", R"({"data":[]})",
    R"({"data":[{"index":0,"embedding":[]}]})",
    R"({"data":[{"index":0,"embedding":["1"]}]})",
    R"({"data":[{"index":0,"embedding":[true]}]})",
    R"({"data":[{"index":-1,"embedding":[1]}]})",
    R"({"data":[{"index":2,"embedding":[1]}]})",
    R"({"data":[{"index":0,"embedding":[1]},{"index":0,"embedding":[2]}]})",
    R"({"data":[{"index":0,"embedding":[1]},{"index":1,"embedding":[2,3]}]})"
  };
  ObOpenAIUtils::ObOpenAIEmbed provider;
  for (const char *input : cases) {
    SCOPED_TRACE(input);
    ObIJsonBase *result = nullptr;
    EXPECT_NE(OB_SUCCESS, provider.parse_output(allocator_, object(input), result));
  }
}

TEST_F(AIProviderTest, nonfinite_vector_element)
{
  ObJsonArray *vectors = nullptr;
  ObJsonArray *vector = nullptr;
  ASSERT_EQ(OB_SUCCESS, ObAIFuncJsonUtils::get_json_array(allocator_, vectors));
  ASSERT_EQ(OB_SUCCESS, ObAIFuncJsonUtils::get_json_array(allocator_, vector));
  auto *number = OB_NEWx(ObJsonDouble, &allocator_, std::numeric_limits<double>::infinity());
  ASSERT_NE(nullptr, number);
  ASSERT_EQ(OB_SUCCESS, vector->append(number));
  ASSERT_EQ(OB_SUCCESS, vectors->append(vector));
  EXPECT_NE(OB_SUCCESS, ObAIFuncUtils::validate_embeddings(allocator_, vectors, 1, 1));
}

TEST_F(AIProviderTest, completion_requires_nonempty_string)
{
  ObOpenAIUtils::ObOpenAIComplete provider;
  const char *cases[] = {R"({})", R"({"choices":[]})",
    R"({"choices":[{"message":{"content":null}}]})",
    R"({"choices":[{"message":{"content":""}}]})",
    R"({"choices":[{"message":{"content":{}}}]})"};
  for (const char *input : cases) {
    ObIJsonBase *result = nullptr;
    EXPECT_NE(OB_SUCCESS, provider.parse_output(allocator_, object(input), result));
  }
}

TEST_F(AIProviderTest, dashscope_completion_rejects_wrong_container_types)
{
  ObDashscopeUtils::ObDashscopeComplete provider;
  const char *cases[] = {
    R"({"output":{"choices":{}}})",
    R"({"output":{"choices":[]}})",
    R"({"output":{"choices":[[]]}})",
    R"({"output":{"choices":[{"message":[]}]}})",
    R"({"output":{"choices":[{"message":{"content":{}}}]}})",
    R"({"output":{"choices":[{"message":{"content":""}}]}})"
  };
  for (const char *input : cases) {
    ObIJsonBase *result = nullptr;
    EXPECT_NE(OB_SUCCESS, provider.parse_output(allocator_, object(input), result));
  }
}

TEST_F(AIProviderTest, ollama_native_request_and_response)
{
  ObAIFuncIComplete *provider = nullptr;
  ASSERT_EQ(OB_SUCCESS, ObAIFuncUtils::get_complete_provider(allocator_, ObString("oLlAmA"), provider));
  ObArray<ObString> headers;
  ObString empty_key;
  ASSERT_EQ(OB_SUCCESS, provider->get_header(allocator_, empty_key, headers));
  EXPECT_EQ(1, headers.count());
  ObString model("test-model"), system("system text"), prompt("user text");
  ObJsonObject *body = nullptr;
  ASSERT_EQ(OB_SUCCESS, provider->get_body(allocator_, model, system, prompt, object(R"({"stream":true})"), body));
  ASSERT_NE(nullptr, body->get_value("stream"));
  EXPECT_FALSE(body->get_value("stream")->get_boolean());
  EXPECT_TRUE(static_cast<ObJsonString *>(body->get_value("prompt"))->get_str() == prompt);
  EXPECT_TRUE(static_cast<ObJsonString *>(body->get_value("system"))->get_str() == system);
  ObIJsonBase *result = nullptr;
  EXPECT_EQ(OB_SUCCESS, provider->parse_output(allocator_, object(R"({"response":"answer"})"), result));
  EXPECT_NE(OB_SUCCESS, provider->parse_output(allocator_, object(R"({"response":null})"), result));
  ObOllamaUtils::ObOllamaEmbed embed;
  EXPECT_EQ(OB_SUCCESS, embed.parse_output(allocator_, object(R"({"embeddings":[[1,2],[3,4]]})"), result));
  EXPECT_NE(OB_SUCCESS, embed.parse_output(allocator_, object(R"({"embeddings":[[1],[3,4]]})"), result));
}

TEST_F(AIProviderTest, rerank_stable_sort_and_validation)
{
  ObJsonArray *ordered = nullptr;
  ObJsonNode *result = object(R"({"results":[{"index":2,"relevance_score":0.5},{"index":1,"relevance_score":1},{"index":0,"relevance_score":0.5}]})")->get_value("results");
  ASSERT_EQ(OB_SUCCESS, ObAIFuncUtils::validate_rerank(allocator_, result, 3, ordered));
  const int64_t expected[] = {1, 0, 2};
  for (int64_t i = 0; i < 3; ++i) {
    EXPECT_EQ(expected[i], static_cast<ObJsonObject *>(ordered->get_value(i))->get_value("index")->get_int());
  }
  EXPECT_NE(OB_SUCCESS, ObAIFuncUtils::validate_rerank(allocator_, result, 4, ordered));
  result = object(R"({"results":[{"index":0,"relevance_score":1},{"index":0,"relevance_score":2}]})")->get_value("results");
  EXPECT_NE(OB_SUCCESS, ObAIFuncUtils::validate_rerank(allocator_, result, 2, ordered));
}
} // namespace
