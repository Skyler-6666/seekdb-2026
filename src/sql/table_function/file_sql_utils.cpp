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

#define USING_LOG_PREFIX SQL_ENG

#include "sql/table_function/file_sql_utils.h"

#include <dirent.h>
#include <errno.h>
#include <math.h>
#include <new>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <algorithm>
#include <utility>

namespace oceanbase
{
namespace sql
{
namespace file_sql
{
namespace
{

const int FILE_SQL_OK = 0;
const int FILE_SQL_EOF = 1;
const int FILE_SQL_ERR = -1;

bool is_space(char c) { return ' ' == c || '\t' == c || '\n' == c || '\r' == c; }

char lower_char(char c)
{
  return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
}

std::string lower_text(const std::string &text)
{
  std::string result = text;
  for (size_t i = 0; i < result.size(); ++i) {
    result[i] = lower_char(result[i]);
  }
  return result;
}

bool ends_with(const std::string &text, const char *suffix)
{
  const size_t suffix_len = strlen(suffix);
  return text.size() >= suffix_len && 0 == text.compare(text.size() - suffix_len, suffix_len, suffix);
}

bool is_digit(char c) { return c >= '0' && c <= '9'; }

bool wildcard_match(const char *pattern, const char *text)
{
  while ('\0' != *pattern) {
    if ('*' == *pattern) {
      while ('*' == *pattern) { ++pattern; }
      if ('\0' == *pattern) { return true; }
      for (const char *p = text; ; ++p) {
        if (wildcard_match(pattern, p)) { return true; }
        if ('\0' == *p) { return false; }
      }
    } else if ('?' == *pattern) {
      if ('\0' == *text) { return false; }
      ++pattern;
      ++text;
    } else {
      if (*pattern != *text) { return false; }
      ++pattern;
      ++text;
    }
  }
  return '\0' == *text;
}

int64_t days_from_civil(int64_t y, int64_t m, int64_t d)
{
  y -= m <= 2 ? 1 : 0;
  const int64_t era = (y >= 0 ? y : y - 399) / 400;
  const int64_t yoe = y - era * 400;
  const int64_t mp = (m + 9) % 12;
  const int64_t doy = (153 * mp + 2) / 5 + d - 1;
  const int64_t doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
  return era * 146097 + doe - 719468;
}

bool valid_date_parts(int64_t y, int64_t m, int64_t d)
{
  static const int days_of_month[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
  if (m < 1 || m > 12 || d < 1) { return false; }
  int max_day = days_of_month[m - 1];
  const bool leap = (0 == y % 4 && 0 != y % 100) || (0 == y % 400);
  if (2 == m && leap) { max_day = 29; }
  return d <= max_day;
}

// Parses yyyy-mm-dd, optionally followed by a single separator and a time part.
bool parse_date_prefix(const std::string &text, int64_t &year, int64_t &month, int64_t &day, size_t &pos)
{
  if (text.size() < 10) { return false; }
  if (!is_digit(text[0]) || !is_digit(text[1]) || !is_digit(text[2]) || !is_digit(text[3]) ||
      '-' != text[4] || !is_digit(text[5]) || !is_digit(text[6]) ||
      '-' != text[7] || !is_digit(text[8]) || !is_digit(text[9])) {
    return false;
  }
  year = (text[0] - '0') * 1000 + (text[1] - '0') * 100 + (text[2] - '0') * 10 + (text[3] - '0');
  month = (text[5] - '0') * 10 + (text[6] - '0');
  day = (text[8] - '0') * 10 + (text[9] - '0');
  pos = 10;
  return valid_date_parts(year, month, day);
}

} // namespace

bool parse_int64(const std::string &text, int64_t &value)
{
  if (text.empty()) { return false; }
  size_t i = 0;
  bool negative = false;
  if ('+' == text[0] || '-' == text[0]) {
    negative = '-' == text[0];
    i = 1;
  }
  if (i >= text.size()) { return false; }
  for (size_t j = i; j < text.size(); ++j) {
    if (!is_digit(text[j])) { return false; }
  }
  if (text.size() - i > 19) { return false; }
  errno = 0;
  char *end = nullptr;
  const long long parsed = strtoll(text.c_str(), &end, 10);
  if (0 != errno || nullptr == end || *end != '\0') { return false; }
  if (negative && parsed > 0) { return false; }
  value = static_cast<int64_t>(parsed);
  return true;
}

bool parse_double(const std::string &text, double &value)
{
  if (text.empty()) { return false; }
  bool seen_digit = false;
  size_t i = 0;
  if ('+' == text[0] || '-' == text[0]) { i = 1; }
  for (size_t j = i; j < text.size(); ++j) {
    const char c = text[j];
    if (is_digit(c)) {
      seen_digit = true;
    } else if ('.' == c || 'e' == c || 'E' == c || '+' == c || '-' == c) {
      // Structural characters are validated by strtod below.
    } else {
      return false;
    }
  }
  if (!seen_digit) { return false; }
  errno = 0;
  char *end = nullptr;
  const double parsed = strtod(text.c_str(), &end);
  if (0 != errno || nullptr == end || *end != '\0') { return false; }
  if (isnan(parsed) || isinf(parsed)) { return false; }
  value = parsed;
  return true;
}

bool parse_boolean(const std::string &text, bool &value)
{
  const std::string lowered = lower_text(text);
  if ("true" == lowered) {
    value = true;
    return true;
  }
  if ("false" == lowered) {
    value = false;
    return true;
  }
  return false;
}

bool parse_date_days(const std::string &text, int32_t &days)
{
  int64_t year = 0;
  int64_t month = 0;
  int64_t day = 0;
  size_t pos = 0;
  if (!parse_date_prefix(text, year, month, day, pos) || pos != text.size()) {
    return false;
  }
  const int64_t value = days_from_civil(year, month, day);
  days = static_cast<int32_t>(value);
  return true;
}

bool parse_datetime_micros(const std::string &text, int64_t &micros)
{
  int64_t year = 0;
  int64_t month = 0;
  int64_t day = 0;
  size_t pos = 0;
  if (!parse_date_prefix(text, year, month, day, pos)) {
    return false;
  }
  if (pos == text.size()) {
    micros = days_from_civil(year, month, day) * 86400LL * 1000000LL;
    return true;
  }
  if ('T' != text[pos] && ' ' != text[pos]) { return false; }
  ++pos;
  if (pos + 8 > text.size()) { return false; }
  if (!is_digit(text[pos]) || !is_digit(text[pos + 1]) || ':' != text[pos + 2] ||
      !is_digit(text[pos + 3]) || !is_digit(text[pos + 4]) || ':' != text[pos + 5] ||
      !is_digit(text[pos + 6]) || !is_digit(text[pos + 7])) {
    return false;
  }
  const int64_t hour = (text[pos] - '0') * 10 + (text[pos + 1] - '0');
  const int64_t minute = (text[pos + 3] - '0') * 10 + (text[pos + 4] - '0');
  const int64_t second = (text[pos + 6] - '0') * 10 + (text[pos + 7] - '0');
  if (hour > 23 || minute > 59 || second > 59) { return false; }
  pos += 8;
  int64_t fraction_micros = 0;
  if (pos < text.size() && '.' == text[pos]) {
    ++pos;
    int digits = 0;
    while (pos < text.size() && is_digit(text[pos]) && digits < 6) {
      fraction_micros = fraction_micros * 10 + (text[pos] - '0');
      ++pos;
      ++digits;
    }
    if (0 == digits) { return false; }
    while (digits < 6) {
      fraction_micros *= 10;
      ++digits;
    }
  }
  if (pos != text.size()) { return false; }
  micros = days_from_civil(year, month, day) * 86400LL * 1000000LL +
           (hour * 3600 + minute * 60 + second) * 1000000LL + fraction_micros;
  return true;
}

bool infer_text_kind(const std::string &text, ColumnKind &kind)
{
  int64_t int_value = 0;
  double double_value = 0.0;
  bool bool_value = false;
  int32_t days = 0;
  int64_t micros = 0;
  if (text.empty()) {
    kind = COLUMN_KIND_VARCHAR;
  } else if (parse_boolean(text, bool_value)) {
    kind = COLUMN_KIND_BOOLEAN;
  } else if (parse_int64(text, int_value)) {
    kind = COLUMN_KIND_BIGINT;
  } else if (parse_date_days(text, days) && 10 == text.size()) {
    kind = COLUMN_KIND_DATE;
  } else if (parse_datetime_micros(text, micros) && text.size() > 10) {
    kind = COLUMN_KIND_DATETIME;
  } else if (parse_double(text, double_value)) {
    kind = COLUMN_KIND_DOUBLE;
  } else {
    kind = COLUMN_KIND_VARCHAR;
  }
  return true;
}

ColumnKind widen_kind(ColumnKind current, ColumnKind next)
{
  if (COLUMN_KIND_NULL == current) { return next; }
  if (COLUMN_KIND_NULL == next) { return current; }
  if (current == next) { return current; }
  if (COLUMN_KIND_VARCHAR == current || COLUMN_KIND_VARCHAR == next) { return COLUMN_KIND_VARCHAR; }
  if ((COLUMN_KIND_BIGINT == current && COLUMN_KIND_DOUBLE == next) ||
      (COLUMN_KIND_DOUBLE == current && COLUMN_KIND_BIGINT == next)) {
    return COLUMN_KIND_DOUBLE;
  }
  if ((COLUMN_KIND_DATE == current && COLUMN_KIND_DATETIME == next) ||
      (COLUMN_KIND_DATETIME == current && COLUMN_KIND_DATE == next)) {
    return COLUMN_KIND_DATETIME;
  }
  return COLUMN_KIND_VARCHAR;
}

const char *kind_name(ColumnKind kind)
{
  switch (kind) {
    case COLUMN_KIND_BIGINT: return "BIGINT";
    case COLUMN_KIND_DOUBLE: return "DOUBLE";
    case COLUMN_KIND_BOOLEAN: return "BOOLEAN";
    case COLUMN_KIND_VARCHAR: return "VARCHAR";
    case COLUMN_KIND_DATE: return "DATE";
    case COLUMN_KIND_DATETIME: return "DATETIME";
    default: return "NULL";
  }
}

std::string format_int64(int64_t value)
{
  char buffer[32];
  snprintf(buffer, sizeof(buffer), "%lld", static_cast<long long>(value));
  return std::string(buffer);
}

std::string format_double(double value)
{
  char buffer[64];
  snprintf(buffer, sizeof(buffer), "%.17g", value);
  return std::string(buffer);
}

int resolve_file_type(const std::string &path, Options &options)
{
  int ret = FILE_SQL_OK;
  if (FILE_TYPE_INVALID != options.type_) {
    // An explicit format argument wins over the suffix.
  } else if (ends_with(path, ".csv")) {
    options.type_ = FILE_TYPE_CSV;
  } else if (ends_with(path, ".jsonl") || ends_with(path, ".ndjson")) {
    options.type_ = FILE_TYPE_JSONL;
  } else if (ends_with(path, ".json")) {
    options.type_ = FILE_TYPE_JSONL;
  } else if (ends_with(path, ".parquet") || ends_with(path, ".pq")) {
    options.type_ = FILE_TYPE_PARQUET;
  } else {
    ret = FILE_SQL_ERR;
  }
  return ret;
}

int check_extension_supported(const std::string &path, const Options &options)
{
  Options copy = options;
  return resolve_file_type(path, copy);
}

int read_file_stamp(const std::string &path, FileStamp &stamp)
{
  int ret = FILE_SQL_OK;
  struct stat file_stat;
  memset(&file_stat, 0, sizeof(file_stat));
  if (0 != ::stat(path.c_str(), &file_stat)) {
    ret = FILE_SQL_ERR;
  } else if (!S_ISREG(file_stat.st_mode)) {
    ret = FILE_SQL_ERR;
  } else {
    stamp.size_ = static_cast<int64_t>(file_stat.st_size);
    stamp.mtime_ = static_cast<int64_t>(file_stat.st_mtime);
  }
  return ret;
}

int expand_glob(const std::string &pattern, std::vector<std::string> &files)
{
  int ret = FILE_SQL_OK;
  files.clear();
  const size_t slash = pattern.find_last_of('/');
  std::string directory = "/";
  std::string base_pattern = pattern;
  if (std::string::npos != slash) {
    directory = 0 == slash ? "/" : pattern.substr(0, slash);
    base_pattern = pattern.substr(slash + 1);
  }
  const bool has_wildcard = std::string::npos != base_pattern.find_first_of("*?");
  if (!has_wildcard) {
    FileStamp stamp;
    if (FILE_SQL_OK == read_file_stamp(pattern, stamp)) {
      files.push_back(pattern);
    }
  } else {
    DIR *dir = ::opendir(directory.c_str());
    if (NULL == dir) {
      ret = FILE_SQL_ERR;
    } else {
      struct dirent *entry = NULL;
      while (NULL != (entry = ::readdir(dir))) {
        const char *name = entry->d_name;
        if (0 == strcmp(name, ".") || 0 == strcmp(name, "..")) { continue; }
        if (!wildcard_match(base_pattern.c_str(), name)) { continue; }
        std::string full = directory;
        if (!full.empty() && '/' != full[full.size() - 1]) { full += "/"; }
        full += name;
        FileStamp stamp;
        if (FILE_SQL_OK == read_file_stamp(full, stamp)) {
          files.push_back(full);
        }
      }
      ::closedir(dir);
      std::sort(files.begin(), files.end());
    }
  }
  return ret;
}

int CsvRowReader::read_char(char &c, bool &eof)
{
  int ret = FILE_SQL_OK;
  eof = false;
  if (pushed_back_) {
    c = push_back_char_;
    pushed_back_ = false;
  } else {
    const int value = ::fgetc(file_);
    if (EOF == value) {
      eof = true;
      ret = FILE_SQL_ERR;
    } else {
      c = static_cast<char>(value);
      if ('\n' == c) { ++line_no_; }
    }
  }
  return ret;
}

int CsvRowReader::open(const std::string &path, const Options &options)
{
  int ret = FILE_SQL_OK;
  close();
  options_ = options;
  header_consumed_ = false;
  line_no_ = 0;
  pushed_back_ = false;
  file_ = ::fopen(path.c_str(), "rb");
  if (NULL == file_) {
    ret = FILE_SQL_ERR;
  } else if (options_.header_) {
    // The header row is consumed on open so that next_row only yields data.
    std::vector<Cell> header;
    bool eof = false;
    ret = scan_record(header, eof);
    header_consumed_ = true;
  }
  return ret;
}

int CsvRowReader::scan_record(std::vector<Cell> &record, bool &eof)
{
  int ret = FILE_SQL_OK;
  eof = false;
  record.clear();
  std::string field;
  bool in_quotes = false;
  bool field_quoted = false;
  bool field_started = false;
  bool record_started = false;
  bool finished = false;
  while (!finished) {
    char c = '\0';
    bool at_eof = false;
    if (FILE_SQL_OK != read_char(c, at_eof)) {
      if (!record_started && field.empty() && !field_started) {
        eof = true;
        ret = FILE_SQL_OK;
      } else {
        Cell cell;
        cell.value_ = field;
        cell.is_null_ = field.empty() && !field_quoted;
        record.push_back(cell);
        ret = FILE_SQL_OK;
      }
      finished = true;
      continue;
    }
    record_started = true;
    if (in_quotes) {
      if (c == options_.quote_) {
        char next = '\0';
        bool next_eof = false;
        if (FILE_SQL_OK == read_char(next, next_eof)) {
          if (next == options_.quote_) {
            field += options_.quote_;
          } else {
            pushed_back_ = true;
            push_back_char_ = next;
            in_quotes = false;
          }
        } else {
          in_quotes = false;
        }
      } else {
        field += c;
      }
    } else if (c == options_.field_term_) {
      Cell cell;
      cell.value_ = field;
      cell.is_null_ = field.empty() && !field_quoted;
      record.push_back(cell);
      field.clear();
      field_quoted = false;
      field_started = false;
    } else if ('\r' == c) {
      char next = '\0';
      bool next_eof = false;
      if (FILE_SQL_OK == read_char(next, next_eof) && '\n' != next) {
        pushed_back_ = true;
        push_back_char_ = next;
      }
      Cell cell;
      cell.value_ = field;
      cell.is_null_ = field.empty() && !field_quoted;
      record.push_back(cell);
      finished = true;
    } else if ('\n' == c) {
      Cell cell;
      cell.value_ = field;
      cell.is_null_ = field.empty() && !field_quoted;
      record.push_back(cell);
      finished = true;
    } else if (c == options_.quote_ && field.empty() && !field_started) {
      in_quotes = true;
      field_quoted = true;
      field_started = true;
    } else {
      field += c;
      field_started = true;
    }
  }
  return ret;
}

int CsvRowReader::next_row(RowValues &row)
{
  int ret = FILE_SQL_OK;
  std::vector<Cell> record;
  bool eof = false;
  row.cells_.clear();
  row.line_no_ = line_no_;
  if (FILE_SQL_OK != (ret = scan_record(record, eof))) {
  } else if (eof) {
    ret = FILE_SQL_EOF;
  } else {
    row.line_no_ = line_no_;
    row.cells_.resize(record.size());
    for (size_t i = 0; i < record.size(); ++i) {
      row.cells_[i].is_null_ = record[i].is_null_;
      row.cells_[i].text_ = record[i].value_;
    }
  }
  return ret;
}

void CsvRowReader::close()
{
  if (NULL != file_) {
    ::fclose(file_);
    file_ = NULL;
  }
  line_no_ = 0;
  pushed_back_ = false;
}

namespace
{

bool parse_json_string(const std::string &line, size_t &pos, std::string &out)
{
  if (pos >= line.size() || '"' != line[pos]) { return false; }
  ++pos;
  out.clear();
  while (pos < line.size()) {
    const char c = line[pos++];
    if ('"' == c) { return true; }
    if ('\\' != c) {
      out += c;
      continue;
    }
    if (pos >= line.size()) { return false; }
    const char escaped = line[pos++];
    switch (escaped) {
      case '"': out += '"'; break;
      case '\\': out += '\\'; break;
      case '/': out += '/'; break;
      case 'b': out += '\b'; break;
      case 'f': out += '\f'; break;
      case 'n': out += '\n'; break;
      case 'r': out += '\r'; break;
      case 't': out += '\t'; break;
      case 'u': {
        if (pos + 4 > line.size()) { return false; }
        unsigned int code = 0;
        for (int i = 0; i < 4; ++i) {
          const char hex = line[pos + i];
          code <<= 4;
          if (hex >= '0' && hex <= '9') { code |= static_cast<unsigned int>(hex - '0'); }
          else if (hex >= 'a' && hex <= 'f') { code |= static_cast<unsigned int>(hex - 'a' + 10); }
          else if (hex >= 'A' && hex <= 'F') { code |= static_cast<unsigned int>(hex - 'A' + 10); }
          else { return false; }
        }
        pos += 4;
        if (code < 0x80) {
          out += static_cast<char>(code);
        } else if (code < 0x800) {
          out += static_cast<char>(0xC0 | (code >> 6));
          out += static_cast<char>(0x80 | (code & 0x3F));
        } else {
          out += static_cast<char>(0xE0 | (code >> 12));
          out += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
          out += static_cast<char>(0x80 | (code & 0x3F));
        }
        break;
      }
      default: return false;
    }
  }
  return false;
}

// Parses one flat JSON value. Nested objects and arrays are rejected.
bool parse_json_value(const std::string &line, size_t &pos, CellValue &cell, ColumnKind &kind)
{
  while (pos < line.size() && is_space(line[pos])) { ++pos; }
  if (pos >= line.size()) { return false; }
  const char c = line[pos];
  if ('"' == c) {
    std::string text;
    if (!parse_json_string(line, pos, text)) { return false; }
    cell.is_null_ = false;
    cell.text_ = text;
    kind = COLUMN_KIND_VARCHAR;
    return true;
  }
  if ('{' == c || '[' == c) {
    // Nested structures are outside the supported JSONL contract.
    return false;
  }
  if ('t' == c) {
    if (line.compare(pos, 4, "true") != 0) { return false; }
    pos += 4;
    cell.is_null_ = false;
    cell.text_ = "true";
    kind = COLUMN_KIND_BOOLEAN;
    return true;
  }
  if ('f' == c) {
    if (line.compare(pos, 5, "false") != 0) { return false; }
    pos += 5;
    cell.is_null_ = false;
    cell.text_ = "false";
    kind = COLUMN_KIND_BOOLEAN;
    return true;
  }
  if ('n' == c) {
    if (line.compare(pos, 4, "null") != 0) { return false; }
    pos += 4;
    cell.is_null_ = true;
    cell.text_.clear();
    kind = COLUMN_KIND_NULL;
    return true;
  }
  size_t begin = pos;
  if ('-' == line[pos] || '+' == line[pos]) { ++pos; }
  bool has_fraction = false;
  while (pos < line.size() && (is_digit(line[pos]) || '.' == line[pos] || 'e' == line[pos] ||
         'E' == line[pos] || '+' == line[pos] || '-' == line[pos])) {
    if ('.' == line[pos] || 'e' == line[pos] || 'E' == line[pos]) { has_fraction = true; }
    ++pos;
  }
  if (pos == begin) { return false; }
  const std::string text = line.substr(begin, pos - begin);
  int64_t int_value = 0;
  double double_value = 0.0;
  cell.is_null_ = false;
  cell.text_ = text;
  if (!has_fraction && parse_int64(text, int_value)) {
    kind = COLUMN_KIND_BIGINT;
  } else if (parse_double(text, double_value)) {
    kind = COLUMN_KIND_DOUBLE;
    cell.text_ = format_double(double_value);
  } else {
    return false;
  }
  return true;
}

bool parse_json_object(const std::string &line, std::vector<std::pair<std::string, CellValue> > &values,
                       std::vector<ColumnKind> &kinds)
{
  size_t pos = 0;
  while (pos < line.size() && is_space(line[pos])) { ++pos; }
  if (pos >= line.size() || '{' != line[pos]) { return false; }
  ++pos;
  values.clear();
  kinds.clear();
  while (true) {
    while (pos < line.size() && is_space(line[pos])) { ++pos; }
    if (pos >= line.size()) { return false; }
    if ('}' == line[pos]) { ++pos; break; }
    std::string key;
    if (!parse_json_string(line, pos, key)) { return false; }
    while (pos < line.size() && is_space(line[pos])) { ++pos; }
    if (pos >= line.size() || ':' != line[pos]) { return false; }
    ++pos;
    CellValue cell;
    ColumnKind kind = COLUMN_KIND_NULL;
    if (!parse_json_value(line, pos, cell, kind)) { return false; }
    values.push_back(std::make_pair(key, cell));
    kinds.push_back(kind);
    while (pos < line.size() && is_space(line[pos])) { ++pos; }
    if (pos < line.size() && ',' == line[pos]) { ++pos; continue; }
    if (pos < line.size() && '}' == line[pos]) { ++pos; break; }
    return false;
  }
  while (pos < line.size() && is_space(line[pos])) { ++pos; }
  return pos == line.size();
}

} // namespace

int JsonlRowReader::open(const std::string &path, const Options &options)
{
  int ret = FILE_SQL_OK;
  close();
  file_ = ::fopen(path.c_str(), "rb");
  if (NULL == file_) {
    ret = FILE_SQL_ERR;
  } else {
    // JSONL has no footer schema; inference reads the file once and the scan
    // then rewinds and streams it row by row.
    std::vector<Column> columns;
    std::string line;
    char buffer[8192];
    bool eof = false;
    while (FILE_SQL_OK == ret && !eof) {
      line.clear();
      bool got_line = false;
      while (NULL != ::fgets(buffer, sizeof(buffer), file_)) {
        line.append(buffer);
        got_line = true;
        if (!line.empty() && '\n' == line[line.size() - 1]) { break; }
      }
      if (!got_line) { eof = true; break; }
      while (!line.empty() && ('\n' == line[line.size() - 1] || '\r' == line[line.size() - 1])) {
        line.erase(line.size() - 1);
      }
      if (line.find_first_not_of(" \t") == std::string::npos) { continue; }
      std::vector<std::pair<std::string, CellValue> > values;
      std::vector<ColumnKind> kinds;
      if (!parse_json_object(line, values, kinds)) {
        ret = FILE_SQL_ERR;
        break;
      }
      for (size_t j = 0; j < values.size(); ++j) {
        size_t index = 0;
        bool found = false;
        for (size_t i = 0; i < columns.size(); ++i) {
          if (columns[i].name_ == values[j].first) { index = i; found = true; break; }
        }
        if (!found) {
          Column column;
          column.name_ = values[j].first;
          columns.push_back(column);
          index = columns.size() - 1;
        }
        columns[index].kind_ = widen_kind(columns[index].kind_, kinds[j]);
        if (COLUMN_KIND_NULL == kinds[j]) { columns[index].nullable_ = true; }
      }
      for (size_t i = 0; i < columns.size(); ++i) {
        bool present = false;
        for (size_t j = 0; j < values.size() && !present; ++j) {
          present = (values[j].first == columns[i].name_);
        }
        if (!present) { columns[i].nullable_ = true; }
      }
    }
    if (FILE_SQL_OK == ret) {
      for (size_t i = 0; i < columns.size(); ++i) {
        if (COLUMN_KIND_NULL == columns[i].kind_) { columns[i].kind_ = COLUMN_KIND_VARCHAR; }
      }
      schema_.columns_.swap(columns);
      if (0 != ::fseek(file_, 0, SEEK_SET)) { ret = FILE_SQL_ERR; }
    }
    (void) options;
  }
  return ret;
}

int JsonlRowReader::next_row(RowValues &row)
{
  int ret = FILE_SQL_OK;
  row.cells_.clear();
  row.line_no_ = 0;
  std::string line;
  bool has_line = false;
  char buffer[8192];
  while (NULL != ::fgets(buffer, sizeof(buffer), file_)) {
    line.append(buffer);
    if (!line.empty() && '\n' == line[line.size() - 1]) {
      line.erase(line.size() - 1);
      if (!line.empty() && '\r' == line[line.size() - 1]) { line.erase(line.size() - 1); }
      has_line = true;
      break;
    }
  }
  if (!has_line) {
    if (!line.empty()) {
      has_line = true;
    }
  }
  if (!has_line) {
    ret = FILE_SQL_EOF;
  } else {
    if (line.find_first_not_of(" \t\r\n") == std::string::npos) {
      // Skip blank lines.
      ret = next_row(row);
    } else {
      std::vector<std::pair<std::string, CellValue> > values;
      std::vector<ColumnKind> kinds;
      if (!parse_json_object(line, values, kinds)) {
        ret = FILE_SQL_ERR;
      } else {
        row.cells_.resize(schema_.columns_.size());
        for (size_t i = 0; i < schema_.columns_.size(); ++i) {
          row.cells_[i].is_null_ = true;
          for (size_t j = 0; j < values.size(); ++j) {
            if (values[j].first == schema_.columns_[i].name_) {
              row.cells_[i] = values[j].second;
              break;
            }
          }
        }
      }
    }
  }
  return ret;
}

void JsonlRowReader::close()
{
  if (NULL != file_) {
    ::fclose(file_);
    file_ = NULL;
  }
}

RowReader *create_row_reader(FileType type)
{
  RowReader *reader = NULL;
  switch (type) {
    case FILE_TYPE_CSV:
      reader = new (std::nothrow) CsvRowReader();
      break;
    case FILE_TYPE_JSONL:
      reader = new (std::nothrow) JsonlRowReader();
      break;
    case FILE_TYPE_PARQUET:
      // Parquet support lands separately; report it as an unsupported format
      // rather than producing silently wrong rows.
      reader = NULL;
      break;
    default:
      break;
  }
  return reader;
}

namespace
{

int infer_csv_schema(const std::string &path, const Options &options, Schema &schema)
{
  int ret = FILE_SQL_OK;
  CsvRowReader reader;
  std::vector<Column> columns;
  if (FILE_SQL_OK != (ret = reader.open(path, options))) {
  } else {
    RowValues row;
    int read_ret = reader.next_row(row);
    if (FILE_SQL_EOF == read_ret) {
      ret = FILE_SQL_OK;
    } else if (FILE_SQL_OK != read_ret) {
      ret = FILE_SQL_ERR;
    } else {
      const size_t column_count = row.cells_.size();
      for (size_t i = 0; i < column_count; ++i) {
        Column column;
        if (options.header_) {
          column.name_ = row.cells_[i].is_null_ ? std::string() : row.cells_[i].text_;
          if (column.name_.empty()) { column.name_ = "column" + format_int64(static_cast<int64_t>(i)); }
          // Duplicate header names get a numeric suffix, matching DuckDB.
          int suffix = 2;
          std::string candidate = column.name_;
          for (size_t j = 0; j < columns.size(); ++j) {
            if (columns[j].name_ == candidate) {
              candidate = column.name_ + "__" + format_int64(suffix++);
              j = static_cast<size_t>(-1);
            }
          }
          column.name_ = candidate;
        } else {
          column.name_ = "column" + format_int64(static_cast<int64_t>(i));
        }
        columns.push_back(column);
      }
      if (options.header_) {
        // The header row is not data.
      } else if (FILE_SQL_OK != (read_ret = reader.next_row(row))) {
        ret = (FILE_SQL_EOF == read_ret) ? FILE_SQL_OK : FILE_SQL_ERR;
      }
      while (FILE_SQL_OK == ret) {
        read_ret = reader.next_row(row);
        if (FILE_SQL_EOF == read_ret) { break; }
        if (FILE_SQL_OK != read_ret) { ret = FILE_SQL_ERR; break; }
        if (row.cells_.size() != column_count) {
          ret = FILE_SQL_ERR;
          break;
        }
        for (size_t i = 0; i < column_count; ++i) {
          if (row.cells_[i].is_null_) {
            columns[i].nullable_ = true;
          } else {
            ColumnKind kind = COLUMN_KIND_NULL;
            infer_text_kind(row.cells_[i].text_, kind);
            columns[i].kind_ = widen_kind(columns[i].kind_, kind);
          }
        }
      }
    }
  }
  if (FILE_SQL_OK == ret) {
    for (size_t i = 0; i < columns.size(); ++i) {
      if (COLUMN_KIND_NULL == columns[i].kind_) {
        // A column that only holds NULLs is reported as VARCHAR so the bound
        // column keeps a usable SQL type; values stay NULL at execution.
        columns[i].kind_ = COLUMN_KIND_VARCHAR;
      }
    }
    schema.columns_.swap(columns);
  }
  reader.close();
  return ret;
}

int infer_jsonl_schema(const std::string &path, const Options &options, Schema &schema)
{
  int ret = FILE_SQL_OK;
  FILE *file = ::fopen(path.c_str(), "rb");
  if (NULL == file) {
    ret = FILE_SQL_ERR;
  } else {
    std::vector<Column> columns;
    std::string line;
    char buffer[8192];
    bool eof = false;
    while (FILE_SQL_OK == ret && !eof) {
      line.clear();
      bool got_line = false;
      while (NULL != ::fgets(buffer, sizeof(buffer), file)) {
        line.append(buffer);
        got_line = true;
        if (!line.empty() && '\n' == line[line.size() - 1]) { break; }
      }
      if (!got_line) { eof = true; break; }
      while (!line.empty() && ('\n' == line[line.size() - 1] || '\r' == line[line.size() - 1])) {
        line.erase(line.size() - 1);
      }
      if (line.find_first_not_of(" \t") == std::string::npos) { continue; }
      std::vector<std::pair<std::string, CellValue> > values;
      std::vector<ColumnKind> kinds;
      if (!parse_json_object(line, values, kinds)) {
        ret = FILE_SQL_ERR;
        break;
      }
      for (size_t j = 0; j < values.size(); ++j) {
        size_t index = 0;
        bool found = false;
        for (size_t i = 0; i < columns.size(); ++i) {
          if (columns[i].name_ == values[j].first) { index = i; found = true; break; }
        }
        if (!found) {
          Column column;
          column.name_ = values[j].first;
          columns.push_back(column);
          index = columns.size() - 1;
        }
        columns[index].kind_ = widen_kind(columns[index].kind_, kinds[j]);
        if (COLUMN_KIND_NULL == kinds[j]) { columns[index].nullable_ = true; }
      }
      // A key that is absent from this record yields NULL for that column.
      for (size_t i = 0; i < columns.size(); ++i) {
        bool present = false;
        for (size_t j = 0; j < values.size() && !present; ++j) {
          present = (values[j].first == columns[i].name_);
        }
        if (!present) { columns[i].nullable_ = true; }
      }
    }
    if (FILE_SQL_OK == ret) {
      for (size_t i = 0; i < columns.size(); ++i) {
        if (COLUMN_KIND_NULL == columns[i].kind_) { columns[i].kind_ = COLUMN_KIND_VARCHAR; }
      }
      schema.columns_.swap(columns);
    }
    ::fclose(file);
  }
  return ret;
}

} // namespace

int infer_schema(const std::string &path, const Options &options, Schema &schema)
{
  int ret = FILE_SQL_OK;
  Options resolved = options;
  if (FILE_SQL_OK != resolve_file_type(path, resolved)) {
    ret = FILE_SQL_ERR;
  } else if (FILE_TYPE_CSV == resolved.type_) {
    ret = infer_csv_schema(path, resolved, schema);
  } else if (FILE_TYPE_JSONL == resolved.type_) {
    ret = infer_jsonl_schema(path, resolved, schema);
  } else if (FILE_TYPE_PARQUET == resolved.type_) {
    ret = FILE_SQL_ERR;
  } else {
    ret = FILE_SQL_ERR;
  }
  return ret;
}

} // namespace file_sql
} // namespace sql
} // namespace oceanbase
