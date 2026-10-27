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

#pragma once

#include <stdint.h>
#include <string>
#include <vector>

namespace oceanbase
{
namespace sql
{
namespace file_sql
{

enum FileType
{
  FILE_TYPE_INVALID = 0,
  FILE_TYPE_CSV = 1,
  FILE_TYPE_JSONL = 2,
  FILE_TYPE_PARQUET = 3
};

// Physical value kinds used by file scans. The names match DuckDB's inference
// vocabulary and are serialized into the scan options string.
enum ColumnKind
{
  COLUMN_KIND_NULL = 0,
  COLUMN_KIND_BIGINT = 1,
  COLUMN_KIND_DOUBLE = 2,
  COLUMN_KIND_BOOLEAN = 3,
  COLUMN_KIND_VARCHAR = 4,
  COLUMN_KIND_DATE = 5,
  COLUMN_KIND_DATETIME = 6
};

struct Options
{
  Options() {}
  FileType type_ = FILE_TYPE_INVALID;
  bool header_ = true;
  int64_t sample_size_ = -1;
  char field_term_ = ',';
  char quote_ = '"';
  char escape_ = '\\';
};

struct Column
{
  std::string name_;
  ColumnKind kind_ = COLUMN_KIND_NULL;
  bool nullable_ = true;
};

struct Schema
{
  std::vector<Column> columns_;
};

// Modification stamp captured when a schema is inferred. Execution re-reads the
// stamp and refuses to scan a file that changed since it was bound.
struct FileStamp
{
  int64_t size_ = -1;
  int64_t mtime_ = -1;
};

struct CellValue
{
  bool is_null_ = true;
  std::string text_;
};

// One row of decoded values in physical column order.
struct RowValues
{
  std::vector<CellValue> cells_;
  int64_t line_no_ = 0;
};

// Derive the file type from the path suffix unless options.type_ was set by an
// explicit format argument.
int resolve_file_type(const std::string &path, Options &options);
int check_extension_supported(const std::string &path, const Options &options);

// Infer the output schema. CSV and JSONL read the whole file (sample_size is
// only allowed to be -1), Parquet reads the footer schema.
int infer_schema(const std::string &path, const Options &options, Schema &schema);

int read_file_stamp(const std::string &path, FileStamp &stamp);

// Expand a glob pattern with a single directory component wildcard set.
int expand_glob(const std::string &pattern, std::vector<std::string> &files);

// Streaming row reader used for execution. Implementations never materialize the
// whole file: they keep one row worth of state plus their input buffer.
class RowReader
{
public:
  virtual ~RowReader() {}
  virtual int open(const std::string &path, const Options &options) = 0;
  // 0 = row produced, 1 = end of file, negative = error.
  virtual int next_row(RowValues &row) = 0;
  virtual void close() = 0;
};

RowReader *create_row_reader(FileType type);

// CSV implementation (exposed for the parser unit tests).
class CsvRowReader : public RowReader
{
public:
  CsvRowReader() {}
  virtual ~CsvRowReader() { close(); }
  int open(const std::string &path, const Options &options) override;
  int next_row(RowValues &row) override;
  void close() override;

private:
  struct Cell
  {
    std::string value_;
    bool is_null_ = false;
  };
  int scan_record(std::vector<Cell> &record, bool &eof);
  int read_char(char &c, bool &eof);

  FILE *file_ = nullptr;
  Options options_;
  bool header_consumed_ = false;
  int64_t line_no_ = 0;
  bool pushed_back_ = false;
  char push_back_char_ = '\0';
};

// JSONL implementation.
class JsonlRowReader : public RowReader
{
public:
  JsonlRowReader() {}
  virtual ~JsonlRowReader() { close(); }
  int open(const std::string &path, const Options &options) override;
  int next_row(RowValues &row) override;
  void close() override;
  // Column order is fixed by the schema inferred at bind time.
  void set_columns(const Schema &schema) { schema_ = schema; }

private:
  FILE *file_ = nullptr;
  Schema schema_;
  bool schema_set_ = false;
};

// Shared flat value parsing helpers (also used by the unit tests).
bool infer_text_kind(const std::string &text, ColumnKind &kind);
ColumnKind widen_kind(ColumnKind current, ColumnKind next);
const char *kind_name(ColumnKind kind);
bool parse_int64(const std::string &text, int64_t &value);
bool parse_double(const std::string &text, double &value);
bool parse_boolean(const std::string &text, bool &value);
bool parse_date_days(const std::string &text, int32_t &days);
bool parse_datetime_micros(const std::string &text, int64_t &micros);
std::string format_double(double value);
std::string format_int64(int64_t value);

} // namespace file_sql
} // namespace sql
} // namespace oceanbase
