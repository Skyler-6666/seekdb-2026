/* Copyright (c) 2026 OceanBase. Licensed under the Apache License, Version 2.0. */
#include "share/ai_service/ai_column_definition.h"
#include <functional>
#include <iostream>
#ifndef AI_COLUMN_STANDALONE_TEST
#include <gtest/gtest.h>
#endif
namespace
{
using namespace oceanbase::share::ai_column;
void check(bool value)
{
  if (!value)
    throw std::runtime_error("AI COLUMN validation assertion failed");
}
void rejects(const std::function<void()> &function)
{
  try
  {
    function();
  }
  catch (const Error &)
  {
    return;
  }
  throw std::runtime_error("AI COLUMN accepted invalid data");
}
void regressions()
{
  Definition d;
  d.expression = "AI_COMPLETE('model',concat('a,b',`content`))";
  d.model = "model";
  d.input_expression = "concat('a,b',`content`)";
  d.config_expression = "NULL";
  d.version = "version-1";
  d.state_column_name = "__ai_s_v1";
  d.request_column_name = "__ai_r_v1";
  d.result_spec_column_name = "__ai_v_v1";
  d.dependencies = {{"content", 17}};
  d.state_column_id = 18;
  d.request_column_id = 19;
  d.result_spec_column_id = 20;
  d.length = 2147483647; // LONGTEXT schema metadata; no allocation uses this bound.
  auto restored = decode(encode(d));
  check(restored.expression == d.expression && restored.dependencies.size() == 1 &&
        restored.dependencies[0].id == 17 && restored.result_spec_column_id == 20 &&
        restored.integer_bits == 64 && restored.length == 2147483647);
  d.length = 0;
  rejects([&] { decode("{}"); });
  rejects([&] { decode("[]"); });
  rejects([&] { decode(encode(d) + "garbage"); });
  check(arguments(d.expression) ==
        std::vector<std::string>({"'model'", "concat('a,b',`content`)"}));
  check(arguments("AI_COMPLETE('x',concat('a''b,c', f(1,2)),json_object('k',3))").size() == 3);
  rejects([] { arguments("AI_COMPLETE('model', concat('x',1)"); });
  check(identifier("a`b") == "`a``b`");
  rejects([] { identifier(std::string("a\0b", 3)); });
  check(literal("'\\\n").find("275c0a") != std::string::npos);
  d.type = ResultType::Integer;
  check(result_literal(d, " -9223372036854775808 ") == "-9223372036854775808");
  rejects([&] { result_literal(d, "9223372036854775808"); });
  rejects([&] { result_literal(d, "1.0"); });
  rejects([&] { result_literal(d, "7 extra"); });
  d.integer_bits = 8;
  check(result_literal(d, "127") == "127");
  check(result_literal(d, "-128") == "-128");
  rejects([&] { result_literal(d, "128"); });
  rejects([&] { result_literal(d, "-129"); });
  d.type = ResultType::Unsigned;
  check(result_literal(d, "255") == "255");
  rejects([&] { result_literal(d, "256"); });
  rejects([&] { result_literal(d, "-1"); });
  d.integer_bits = 64;
  check(result_literal(d, "18446744073709551615") == "18446744073709551615");
  rejects([&] { result_literal(d, "18446744073709551616"); });
  d.type = ResultType::Double;
  check(result_literal(d, "1.25e3") == "1.25e3");
  rejects([&] { result_literal(d, "1e309"); });
  rejects([&] { result_literal(d, "NaN"); });
  d.type = ResultType::Json;
  check(result_literal(d, " {\"k\":[1,true,null]} ").find("AS JSON") != std::string::npos);
  check(!result_literal(d, "null").empty());
  rejects([&] { result_literal(d, "```json\n{}\n```"); });
  rejects([&] { result_literal(d, "{bad}"); });
  d.type = ResultType::Vector;
  d.dimension = 4;
  check(!result_literal(d, "[0,1.5,-2,3]").empty());
  for (const char *bad :
       {"[1,2,3]", "[1,2,3,4,5]", "[1,null,3,4]", "[1,\"2\",3,4]", "[1,2,3,1e100]", "{}"})
    rejects([&] { result_literal(d, bad); });
  d.type = ResultType::Date;
  check(!result_literal(d, "2024-02-29").empty());
  rejects([&] { result_literal(d, "2023-02-29"); });
  rejects([&] { result_literal(d, "2024-13-01"); });
  d.type = ResultType::Datetime;
  check(!result_literal(d, "2024-02-29 23:59:59.123456").empty());
  rejects([&] { result_literal(d, "2024-02-29 23:59:59."); });
  rejects([&] { result_literal(d, "2024-02-29 24:00:00"); });
  d.type = ResultType::Text;
  d.length = 2;
  check(!result_literal(d, "你好").empty());
  rejects([&] { result_literal(d, "你好呀"); });
  rejects([&] { result_literal(d, std::string("\xc0\xaf", 2)); });
  rejects([&] { result_literal(d, std::string("\xed\xa0\x80", 3)); });
  check(instruction(ResultType::Json).find("Return exactly one valid JSON value") !=
        std::string::npos);
  check(instruction(ResultType::Integer).find("base-10 integer") != std::string::npos);
  std::cout << "PASS: AI COLUMN definition round-trip and typed result validation" << std::endl;
}
} // namespace
#ifdef AI_COLUMN_STANDALONE_TEST
int main()
{
  try
  {
    regressions();
    return 0;
  }
  catch (const std::exception &error)
  {
    std::cerr << error.what() << std::endl;
    return 1;
  }
}
#else
TEST(AIColumnResolver, TypedResultsAndStoredDefinition)
{
  regressions();
}
#endif
