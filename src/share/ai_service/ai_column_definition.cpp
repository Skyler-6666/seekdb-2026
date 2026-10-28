/* Copyright (c) 2026 OceanBase. Licensed under the Apache License, Version 2.0. */
#include "ai_column_definition.h"
#include <algorithm>
#include <atomic>
#include <charconv>
#include <cmath>
#include <limits>
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

namespace oceanbase
{
namespace share
{
namespace ai_column
{
namespace
{
[[noreturn]] void invalid(const std::string &message)
{
  throw Error(-4002, "AI COLUMN: " + message);
}
std::string trim(std::string value)
{
  size_t begin = value.find_first_not_of(" \r\n\t");
  if (begin == std::string::npos)
    return "";
  return value.substr(begin, value.find_last_not_of(" \r\n\t") - begin + 1);
}
std::string compact(const rapidjson::Value &value)
{
  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  value.Accept(writer);
  return std::string(buffer.GetString(), buffer.GetSize());
}
bool digits(const std::string &text, size_t at, size_t count)
{
  if (at + count > text.size())
    return false;
  for (size_t i = at; i < at + count; ++i)
    if (text[i] < '0' || text[i] > '9')
      return false;
  return true;
}
int number(const std::string &text, size_t at, size_t count)
{
  int value = 0;
  for (size_t i = at; i < at + count; ++i)
    value = value * 10 + text[i] - '0';
  return value;
}
bool date(const std::string &text)
{
  if (text.size() < 10 || !digits(text, 0, 4) || !digits(text, 5, 2) || !digits(text, 8, 2) ||
      text[4] != '-' || text[7] != '-')
    return false;
  int year = number(text, 0, 4), month = number(text, 5, 2), day = number(text, 8, 2);
  const int days[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
  return year > 0 && month > 0 && month <= 12 && day > 0 &&
         day <= days[month - 1] +
                    (month == 2 && year % 4 == 0 && (year % 100 != 0 || year % 400 == 0));
}
} // namespace
std::string identifier(const std::string &name)
{
  if (name.empty() || name.find('\0') != std::string::npos)
    invalid("invalid SQL identifier");
  std::string result = "`";
  for (char c : name)
  {
    if (c == '`')
      result += '`';
    result += c;
  }
  return result + '`';
}
std::string literal(const std::string &value)
{
  // Hex literals are independent of NO_BACKSLASH_ESCAPES and connection escaping.
  static const char hex[] = "0123456789abcdef";
  std::string result = "CONVERT(X'";
  result.reserve(value.size() * 2 + 32);
  for (unsigned char c : value)
  {
    result += hex[c >> 4];
    result += hex[c & 15];
  }
  return result + "' USING utf8mb4)";
}
std::vector<std::string> arguments(const std::string &expression)
{
  size_t start = expression.find('('), end = expression.find_last_of(')');
  if (start == std::string::npos || end == std::string::npos || end <= start)
    invalid("expected a function call");
  std::vector<std::string> values;
  char quote = 0;
  unsigned depth = 0;
  size_t begin = start + 1;
  for (size_t i = begin; i < end; ++i)
  {
    char c = expression[i];
    if (quote)
    {
      if (c == '\\' && quote != '`')
      {
        if (i + 1 < end)
          ++i;
      }
      else if (c == quote)
      {
        if (i + 1 < end && expression[i + 1] == quote)
          ++i;
        else
          quote = 0;
      }
    }
    else if (c == '\'' || c == '"' || c == '`')
      quote = c;
    else if (c == '(' || c == '[')
      ++depth;
    else if (c == ')' || c == ']')
    {
      if (depth == 0)
        invalid("unbalanced expression");
      --depth;
    }
    else if (c == ',' && depth == 0)
    {
      values.push_back(trim(expression.substr(begin, i - begin)));
      begin = i + 1;
    }
  }
  if (quote || depth)
    invalid("unterminated expression");
  values.push_back(trim(expression.substr(begin, end - begin)));
  return values;
}
std::string instruction(ResultType type)
{
  switch (type)
  {
  case ResultType::Json:
    return "Return exactly one valid JSON value. Do not include markdown or explanatory text.";
  case ResultType::Integer:
    return "Return exactly one base-10 integer in the signed 64-bit range. Do not include other "
           "text.";
  case ResultType::Unsigned:
    return "Return exactly one nonnegative base-10 integer in the unsigned 64-bit range. Do not "
           "include other text.";
  case ResultType::Double:
    return "Return exactly one finite numeric value. Do not include other text.";
  case ResultType::Boolean:
    return "Return exactly one boolean value: true or false.";
  case ResultType::Date:
    return "Return exactly one valid date in YYYY-MM-DD format.";
  case ResultType::Datetime:
    return "Return exactly one valid datetime in YYYY-MM-DD HH:MM:SS format.";
  case ResultType::Text:
    return "Return only the requested text value, without markdown fences or explanations.";
  case ResultType::Vector:
    return "";
  }
  invalid("invalid result type");
}
std::string encode(const Definition &d)
{
  rapidjson::StringBuffer b;
  rapidjson::Writer<rapidjson::StringBuffer> w(b);
  w.StartObject();
  auto text = [&](const char *key, const std::string &value)
  {
    w.Key(key);
    w.String(value.data(), value.size());
  };
  auto integer = [&](const char *key, uint64_t value)
  {
    w.Key(key);
    w.Uint64(value);
  };
  text("expression", d.expression);
  text("normalized_expr", d.normalized_expression);
  text("model", d.model);
  text("input_expression", d.input_expression);
  text("config_expression", d.config_expression);
  text("version", d.version);
  text("type_prompt", d.type_prompt);
  text("state_name", d.state_column_name);
  text("request_name", d.request_column_name);
  text("spec_name", d.result_spec_column_name);
  integer("type", static_cast<unsigned>(d.type));
  integer("max_retries", d.max_retries);
  integer("dimension", d.dimension);
  integer("integer_bits", d.integer_bits);
  integer("length", d.length);
  integer("scale", d.scale);
  integer("state_column_id", d.state_column_id);
  integer("request_column_id", d.request_column_id);
  integer("result_spec_column_id", d.result_spec_column_id);
  w.Key("embedding");
  w.Bool(d.embedding);
  w.Key("dependencies");
  w.StartArray();
  for (const auto &dep : d.dependencies)
  {
    w.StartObject();
    text("name", dep.name);
    integer("id", dep.id);
    w.EndObject();
  }
  w.EndArray();
  w.EndObject();
  return std::string(b.GetString(), b.GetSize());
}
Definition decode(const std::string &value)
{
  rapidjson::Document doc;
  doc.Parse<rapidjson::kParseValidateEncodingFlag>(value.data(), value.size());
  if (doc.HasParseError() || !doc.IsObject())
    invalid("invalid stored definition");
  Definition d;
  auto text = [&](const char *key)
  {
    if (!doc.HasMember(key) || !doc[key].IsString())
      invalid("missing definition field");
    return std::string(doc[key].GetString(), doc[key].GetStringLength());
  };
  auto integer = [&](const char *key)
  {
    if (!doc.HasMember(key) || !doc[key].IsUint64())
      invalid("invalid definition field");
    return doc[key].GetUint64();
  };
  d.expression = text("expression");
  d.normalized_expression = text("normalized_expr");
  d.model = text("model");
  d.input_expression = text("input_expression");
  d.config_expression = text("config_expression");
  d.version = text("version");
  d.type_prompt = text("type_prompt");
  d.state_column_name = text("state_name");
  d.request_column_name = text("request_name");
  d.result_spec_column_name = text("spec_name");
  uint64_t type = integer("type"), retries = integer("max_retries"),
           dimension = integer("dimension"), length = integer("length"), scale = integer("scale");
  if (type > static_cast<unsigned>(ResultType::Vector) || retries > 100 || dimension > 65536 ||
      length > 64 * 1024 * 1024 || scale > 30)
    invalid("definition bounds");
  d.integer_bits = integer("integer_bits");
  if (d.integer_bits != 8 && d.integer_bits != 16 && d.integer_bits != 24 && d.integer_bits != 32 &&
      d.integer_bits != 64)
    invalid("invalid integer width");
  d.type = static_cast<ResultType>(type);
  d.max_retries = retries;
  d.dimension = dimension;
  d.length = length;
  d.scale = scale;
  d.state_column_id = integer("state_column_id");
  d.request_column_id = integer("request_column_id");
  d.result_spec_column_id = integer("result_spec_column_id");
  if (!doc.HasMember("embedding") || !doc["embedding"].IsBool() || !doc.HasMember("dependencies") ||
      !doc["dependencies"].IsArray() || doc["dependencies"].Size() > 1024)
    invalid("invalid dependencies");
  d.embedding = doc["embedding"].GetBool();
  for (const auto &dep : doc["dependencies"].GetArray())
  {
    if (!dep.IsObject() || !dep.HasMember("name") || !dep["name"].IsString() ||
        !dep.HasMember("id") || !dep["id"].IsUint64())
      invalid("invalid dependency");
    d.dependencies.push_back({std::string(dep["name"].GetString(), dep["name"].GetStringLength()),
                              dep["id"].GetUint64()});
  }
  return d;
}
std::string result_literal(const Definition &d, const std::string &raw)
{
  if (raw.size() > MAX_RESULT_BYTES)
    invalid("model result exceeds 1 MiB");
  std::string value = trim(raw);
  switch (d.type)
  {
  case ResultType::Text:
  {
    size_t characters = 0;
    for (size_t i = 0; i < raw.size(); ++characters)
    {
      unsigned char first = raw[i++];
      unsigned count = 0;
      uint32_t point = first;
      if (first >= 0x80)
      {
        if (first >= 0xc2 && first <= 0xdf)
        {
          count = 1;
          point = first & 31;
        }
        else if (first >= 0xe0 && first <= 0xef)
        {
          count = 2;
          point = first & 15;
        }
        else if (first >= 0xf0 && first <= 0xf4)
        {
          count = 3;
          point = first & 7;
        }
        else
          invalid("invalid UTF-8 result");
        if (raw.size() - i < count)
          invalid("incomplete UTF-8 result");
        for (unsigned j = 0; j < count; ++j)
        {
          unsigned char next = raw[i++];
          if ((next & 0xc0) != 0x80)
            invalid("invalid UTF-8 result");
          point = (point << 6) | (next & 63);
        }
        if ((count == 2 && point < 0x800) || (count == 3 && point < 0x10000) || point > 0x10ffff ||
            (point >= 0xd800 && point <= 0xdfff))
          invalid("invalid UTF-8 code point");
      }
    }
    if (d.length > 0 && characters > static_cast<size_t>(d.length))
      invalid("text result exceeds column length");
    return literal(raw);
  }
  case ResultType::Json:
  case ResultType::Vector:
  {
    rapidjson::Document parsed;
    parsed.Parse<rapidjson::kParseValidateEncodingFlag | rapidjson::kParseFullPrecisionFlag>(
        value.data(), value.size());
    if (parsed.HasParseError())
      invalid("model result is not valid JSON");
    if (d.type == ResultType::Vector)
    {
      if (!parsed.IsArray() || parsed.Size() != static_cast<unsigned>(d.dimension))
        throw Error(-7600, "AI COLUMN: vector dimension mismatch");
      for (auto &v : parsed.GetArray())
        if (!v.IsNumber() || !std::isfinite(v.GetDouble()) ||
            std::abs(v.GetDouble()) > std::numeric_limits<float>::max())
          invalid("invalid vector element");
      return literal(compact(parsed));
    }
    return "CAST(" + literal(compact(parsed)) + " AS JSON)";
  }
  case ResultType::Integer:
  {
    int64_t number = 0;
    auto result = std::from_chars(value.data(), value.data() + value.size(), number);
    if (result.ec != std::errc() || result.ptr != value.data() + value.size() || value.empty())
      invalid("invalid signed integer result");
    if (d.integer_bits < 64 && (number < -(int64_t(1) << (d.integer_bits - 1)) ||
                                number > (int64_t(1) << (d.integer_bits - 1)) - 1))
      invalid("integer result outside column range");
    return std::to_string(number);
  }
  case ResultType::Unsigned:
  {
    uint64_t number = 0;
    auto result = std::from_chars(value.data(), value.data() + value.size(), number);
    if (result.ec != std::errc() || result.ptr != value.data() + value.size() || value.empty())
      invalid("invalid unsigned integer result");
    if (d.integer_bits < 64 && number > ((uint64_t(1) << d.integer_bits) - 1))
      invalid("integer result outside column range");
    return std::to_string(number);
  }
  case ResultType::Double:
  {
    double number = 0;
    auto result = std::from_chars(value.data(), value.data() + value.size(), number);
    if (result.ec != std::errc() || result.ptr != value.data() + value.size() ||
        !std::isfinite(number) || value.empty())
      invalid("invalid numeric result");
    return value;
  }
  case ResultType::Boolean:
    if (value == "true" || value == "1")
      return "1";
    if (value == "false" || value == "0")
      return "0";
    invalid("invalid boolean result");
  case ResultType::Date:
    if (value.size() != 10 || !date(value))
      invalid("invalid DATE result");
    return literal(value);
  case ResultType::Datetime:
    if (value.size() < 19 || value.size() > 26 || !date(value) ||
        (value[10] != ' ' && value[10] != 'T') || value[13] != ':' || value[16] != ':' ||
        !digits(value, 11, 2) || !digits(value, 14, 2) || !digits(value, 17, 2) ||
        number(value, 11, 2) > 23 || number(value, 14, 2) > 59 || number(value, 17, 2) > 59 ||
        (value.size() > 19 &&
         (value.size() == 20 || value[19] != '.' || !digits(value, 20, value.size() - 20))))
      invalid("invalid DATETIME result");
    return literal(value);
  }
  invalid("unsupported result type");
}
} // namespace ai_column
} // namespace share
} // namespace oceanbase
