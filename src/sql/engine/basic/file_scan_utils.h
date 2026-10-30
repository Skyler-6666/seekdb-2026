/* Copyright (c) 2026 OceanBase. Licensed under the Apache License, Version 2.0. */
#pragma once

#include <cstdio>
#include <cstdint>
#include <memory>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace oceanbase { namespace sql { namespace file_sql {

enum class Type { Null, Bigint, Double, Boolean, Varchar, Date, Datetime };
struct Column {
  std::string name;
  std::string key;
  Type type = Type::Null;
  bool nullable = false;
  int scale = 0;
};
struct Options {
  bool header = true;
  char delimiter = ',';
  char quote = '"';
  char escape = '"';
  std::string null_text;
};
struct Plan {
  std::string format;
  std::string path;
  Options options;
  std::vector<Column> columns;
  std::string fingerprint;
  uint64_t rows = 0;
};
using Cell = std::optional<std::string>;
using Row = std::vector<Cell>;

class FileSqlError : public std::runtime_error {
public:
  explicit FileSqlError(const std::string &message) : std::runtime_error(message) {}
};

Type infer_type(const std::string &value);
Type widen(Type left, Type right);
const char *type_name(Type type);
std::string checked_path(const std::string &path, const std::string &secure_root);
std::vector<std::string> discover(const std::string &pattern, const std::string &secure_root);
Plan infer(Plan source, const std::string &secure_root, std::function<void()> check = {});
std::string encode(const Plan &plan);
Plan decode(const std::string &json);

class ParquetReader;

// Owns one input handle and one record. Schema inference never retains rows.
class Reader {
public:
  Reader(const Plan &plan, const std::string &secure_root, std::function<void()> check = {});
  ~Reader();
  Reader(const Reader &) = delete;
  Reader &operator=(const Reader &) = delete;
  bool next(Row &row);
  void rescan();
  void project(const std::vector<size_t> &columns);
  void check_unchanged() const;
private:
  friend Plan infer(Plan, const std::string &, std::function<void()>);
  bool csv(Row &row);
  bool json(Row &row, std::vector<std::string> &keys);
  bool raw(Row &row, std::vector<std::string> &keys);
  int get();
  void fail(const std::string &message) const;
  std::string stamp() const;
  std::function<void()> check_;
  uint64_t records_checked_ = 0;
  Plan plan_;
  std::string secure_root_;
  std::string fingerprint_;
  std::unique_ptr<FILE, int(*)(FILE *)> file_;
  uint64_t line_ = 1;
  uint64_t record_line_ = 1;
  bool header_pending_ = true;
  bool first_record_ = true;
  std::unique_ptr<ParquetReader> parquet_;
  std::vector<std::string> glob_files_;
  std::vector<Type> json_types_;
  size_t glob_index_ = 0;
};

}}} // namespace oceanbase::sql::file_sql
