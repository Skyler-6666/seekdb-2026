/* Copyright (c) 2026 OceanBase. Licensed under the Apache License, Version 2.0. */
#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace oceanbase
{
namespace share
{
namespace ai_column
{
inline constexpr size_t MAX_RESULT_BYTES = 1024 * 1024;
inline constexpr size_t MAX_GROUP_INPUT_BYTES = 16 * 1024 * 1024;
inline constexpr size_t MAX_COLUMNS = 16;
struct Error : std::runtime_error
{
  int code;
  Error(int code, const std::string &message) : std::runtime_error(message), code(code) {}
};
enum class ResultType
{
  Text,
  Json,
  Integer,
  Unsigned,
  Double,
  Boolean,
  Date,
  Datetime,
  Vector
};
struct Dependency
{
  std::string name;
  uint64_t id = 0;
};
struct Definition
{
  std::string expression, normalized_expression, model, input_expression, config_expression;
  std::string version, type_prompt;
  std::string state_column_name, request_column_name, result_spec_column_name;
  std::vector<Dependency> dependencies;
  ResultType type = ResultType::Text;
  uint64_t state_column_id = 0, request_column_id = 0, result_spec_column_id = 0;
  int max_retries = 2, dimension = 0, length = 0, scale = 0, integer_bits = 64;
  bool embedding = false;
};
std::string encode(const Definition &definition);
Definition decode(const std::string &text);
std::vector<std::string> arguments(const std::string &expression);
std::string identifier(const std::string &name);
std::string literal(const std::string &value);
std::string instruction(ResultType type);
// Returns a validated SQL literal; SQL NULL is represented separately by the caller.
std::string result_literal(const Definition &definition, const std::string &value);
} // namespace ai_column
} // namespace share
} // namespace oceanbase
