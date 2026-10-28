/* Copyright (c) 2026 OceanBase. Licensed under the Apache License, Version 2.0. */
#include "file_scan_utils.h"
#include "file_parquet_reader.h"
#include <algorithm>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <set>
#include <system_error>
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace oceanbase { namespace sql { namespace file_sql {
namespace fs = std::filesystem;
namespace {
constexpr size_t MAX_RECORD = 16 * 1024 * 1024;
constexpr size_t MAX_COLUMNS = 1024;
constexpr size_t MAX_FILES = 10000;
fs::path from_utf8(const std::string &value)
{
  return fs::path(std::u8string(reinterpret_cast<const char8_t *>(value.data()), value.size()));
}
std::string utf8_path(const fs::path &path)
{
  auto value = path.generic_u8string();
  return std::string(reinterpret_cast<const char *>(value.data()), value.size());
}
bool valid_utf8(const std::string &value)
{
  for (size_t i = 0; i < value.size();) {
    uint32_t point = static_cast<unsigned char>(value[i++]);
    if (point < 0x80) continue;
    size_t rest = point >= 0xc2 && point <= 0xdf ? 1 : point >= 0xe0 && point <= 0xef ? 2 : point >= 0xf0 && point <= 0xf4 ? 3 : 0;
    if (rest == 0 || i + rest > value.size()) return false;
    uint32_t minimum = rest == 1 ? 0x80 : rest == 2 ? 0x800 : 0x10000;
    point &= (1u << (6 - rest)) - 1;
    for (size_t n = 0; n < rest; ++n) {
      unsigned char ch = static_cast<unsigned char>(value[i++]);
      if ((ch & 0xc0) != 0x80) return false;
      point = (point << 6) | (ch & 0x3f);
    }
    if (point < minimum || point > 0x10ffff || (point >= 0xd800 && point <= 0xdfff)) return false;
  }
  return true;
}
void skip_bom(FILE *file)
{
  unsigned char bytes[3];
  size_t count = fread(bytes, 1, sizeof(bytes), file);
  if (count != 3 || bytes[0] != 0xef || bytes[1] != 0xbb || bytes[2] != 0xbf) {
    if (fseek(file, 0, SEEK_SET) != 0) throw FileSqlError("file SQL: cannot rewind input");
  }
}
std::string lower(std::string value)
{
  for (char &c : value) { if (c >= 'A' && c <= 'Z') c += 'a' - 'A'; }
  return value;
}
bool inside(const fs::path &path, const fs::path &root)
{
  auto p = path.begin();
  for (auto r = root.begin(); r != root.end(); ++r, ++p) {
    if (p == path.end()) return false;
#ifdef _WIN32
    if (lower(utf8_path(*p)) != lower(utf8_path(*r))) return false;
#else
    if (*p != *r) return false;
#endif
  }
  return true;
}
bool digits(const std::string &s, size_t start, size_t length)
{
  if (start + length > s.size()) return false;
  for (size_t i = start; i < start + length; ++i) if (s[i] < '0' || s[i] > '9') return false;
  return true;
}
int decimal(const std::string &s, size_t at, size_t count)
{
  int result = 0;
  for (size_t i = at; i < at + count; ++i) result = result * 10 + s[i] - '0';
  return result;
}
bool date(const std::string &s)
{
  if (s.size() < 10 || s[4] != '-' || s[7] != '-' || !digits(s, 0, 4)
      || !digits(s, 5, 2) || !digits(s, 8, 2)) return false;
  int year = decimal(s, 0, 4), month = decimal(s, 5, 2), day = decimal(s, 8, 2);
  const int days[] = {31,28,31,30,31,30,31,31,30,31,30,31};
  if (year == 0 || month == 0 || month > 12 || day == 0) return false;
  return day <= days[month - 1] + (month == 2 && year % 4 == 0 && (year % 100 != 0 || year % 400 == 0));
}
bool datetime(const std::string &s)
{
  if (s.size() < 19 || !date(s) || (s[10] != 'T' && s[10] != ' ') || s[13] != ':' || s[16] != ':'
      || !digits(s, 11, 2) || !digits(s, 14, 2) || !digits(s, 17, 2)
      || decimal(s, 11, 2) > 23 || decimal(s, 14, 2) > 59 || decimal(s, 17, 2) > 59) return false;
  return s.size() == 19 || (s.size() >= 21 && s.size() <= 26 && s[19] == '.' && digits(s, 20, s.size() - 20));
}
bool wildcard(const std::string &pattern, const std::string &name)
{
  size_t p = 0, n = 0, star = std::string::npos, retry = 0;
  while (n < name.size()) {
    if (p < pattern.size() && (pattern[p] == '?' || pattern[p] == name[n])) { ++p; ++n; }
    else if (p < pattern.size() && pattern[p] == '*') { star = p++; retry = n; }
    else if (star != std::string::npos) { p = star + 1; n = ++retry; }
    else return false;
  }
  while (p < pattern.size() && pattern[p] == '*') ++p;
  return p == pattern.size();
}
void unique_names(std::vector<Column> &columns)
{
  std::set<std::string> used;
  for (size_t i = 0; i < columns.size(); ++i) {
    std::string base = columns[i].name.empty() ? "column" + std::to_string(i) : columns[i].name;
    std::string name = base;
    size_t duplicate = 2;
    while (!used.insert(lower(name)).second) name = base + "__" + std::to_string(duplicate++);
    if (name.size() > 256) throw FileSqlError("file SQL: column name exceeds 256 bytes");
    columns[i].name = name;
  }
}
std::string scalar(const rapidjson::Value &value)
{
  if (value.IsString()) return std::string(value.GetString(), value.GetStringLength());
  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  value.Accept(writer);
  return std::string(buffer.GetString(), buffer.GetSize());
}
}

Type infer_type(const std::string &value)
{
  if (value.empty()) return Type::Varchar;
  std::string folded = lower(value);
  if (folded == "true" || folded == "false") return Type::Boolean;
  size_t start = value[0] == '-' || value[0] == '+' ? 1 : 0;
  if (start < value.size() && digits(value, start, value.size() - start)) {
    int64_t number = 0;
    const char *begin = value.data() + (value[0] == '+' ? 1 : 0);
    auto result = std::from_chars(begin, value.data() + value.size(), number);
    return result.ec == std::errc() ? Type::Bigint : Type::Varchar;
  }
  double number = 0;
  const char *begin = value.data() + (value[0] == '+' ? 1 : 0);
  auto result = std::from_chars(begin, value.data() + value.size(), number);
  if (result.ec == std::errc() && result.ptr == value.data() + value.size() && std::isfinite(number)) return Type::Double;
  if (value.size() == 10 && date(value)) return Type::Date;
  if (datetime(value)) return Type::Datetime;
  return Type::Varchar;
}
Type widen(Type left, Type right)
{
  if (left == Type::Null) return right;
  if (right == Type::Null || left == right) return left;
  if ((left == Type::Bigint && right == Type::Double) || (left == Type::Double && right == Type::Bigint)) return Type::Double;
  if ((left == Type::Date && right == Type::Datetime) || (left == Type::Datetime && right == Type::Date)) return Type::Datetime;
  return Type::Varchar;
}
const char *type_name(Type type)
{
  const char *names[] = {"NULL", "BIGINT", "DOUBLE", "BOOLEAN", "VARCHAR", "DATE", "DATETIME"};
  return names[static_cast<size_t>(type)];
}
std::string checked_path(const std::string &path, const std::string &secure_root)
{
  if (path.empty() || path.find('\0') != std::string::npos || secure_root.empty()
      || lower(secure_root) == "null" || secure_root.find('\0') != std::string::npos)
    throw FileSqlError("file SQL: access denied by secure_file_priv");
  std::error_code error;
  fs::path root = fs::canonical(from_utf8(secure_root), error);
  if (error || !fs::is_directory(root)) throw FileSqlError("file SQL: secure_file_priv is not an existing directory");
  fs::path target = fs::canonical(from_utf8(path), error);
  if (error) throw FileSqlError("file SQL: cannot resolve input path: " + path);
  if (!inside(target, root)) throw FileSqlError("file SQL: path escapes secure_file_priv: " + path);
  return utf8_path(target);
}
std::vector<std::string> discover(const std::string &pattern, const std::string &secure_root)
{
  fs::path path = from_utf8(pattern);
  std::string parent = checked_path(path.has_parent_path() ? utf8_path(path.parent_path()) : ".", secure_root);
  std::vector<std::string> files;
  std::error_code error;
  fs::directory_iterator entries(from_utf8(parent), error);
  if (error) throw FileSqlError("file SQL: cannot list directory: " + parent);
  for (const auto &entry : entries) {
    if (wildcard(utf8_path(path.filename()), utf8_path(entry.path().filename())) && entry.is_regular_file()) {
      files.push_back(checked_path(utf8_path(entry.path()), secure_root));
      if (files.size() > MAX_FILES) throw FileSqlError("file SQL: glob exceeds 10000 files");
    }
  }
  std::sort(files.begin(), files.end());
  return files;
}

Reader::Reader(const Plan &plan, const std::string &secure_root, std::function<void()> check)
  : check_(std::move(check)), plan_(plan), secure_root_(secure_root), file_(nullptr, fclose), header_pending_(plan.options.header)
{
  if (plan_.format == "glob") { glob_files_ = discover(plan.path, secure_root); return; }
  plan_.path = checked_path(plan.path, secure_root);
  if (!fs::is_regular_file(from_utf8(plan_.path))) fail("input is not a regular file");
#ifdef _WIN32
  HANDLE handle = CreateFileW(from_utf8(plan_.path).c_str(), GENERIC_READ,
      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
  if (handle == INVALID_HANDLE_VALUE) fail("cannot open input");
  int descriptor = _open_osfhandle(reinterpret_cast<intptr_t>(handle), _O_RDONLY | _O_BINARY);
  if (descriptor < 0) { CloseHandle(handle); fail("cannot attach input descriptor"); }
  FILE *opened = _fdopen(descriptor, "rb");
  if (opened == nullptr) { _close(descriptor); fail("cannot attach input stream"); }
  file_.reset(opened);
  std::wstring final_path(32768, L'\0');
  DWORD size = GetFinalPathNameByHandleW(handle, final_path.data(), static_cast<DWORD>(final_path.size()), FILE_NAME_NORMALIZED);
  if (size == 0 || size >= final_path.size()) fail("cannot verify opened file path");
  final_path.resize(size);
  if (final_path.compare(0, 4, L"\\\\?\\") == 0) final_path.erase(0, 4);
  checked_path(utf8_path(fs::path(final_path)), secure_root);
#else
  int descriptor = ::open(plan_.path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (descriptor < 0) fail("cannot open input");
  FILE *opened = fdopen(descriptor, "rb");
  if (opened == nullptr) { ::close(descriptor); fail("cannot attach input stream"); }
  file_.reset(opened);
#if defined(__linux__)
  char final_path[4096];
  std::string proc = "/proc/self/fd/" + std::to_string(descriptor);
  ssize_t length = readlink(proc.c_str(), final_path, sizeof(final_path));
  if (length <= 0 || static_cast<size_t>(length) == sizeof(final_path)) fail("cannot verify opened file path");
  checked_path(std::string(final_path, static_cast<size_t>(length)), secure_root);
#endif
#endif
  fingerprint_ = stamp();
  if (!plan.fingerprint.empty() && plan.fingerprint != fingerprint_) fail("input changed after schema inference");
  if (plan.format == "csv" || plan.format == "json") skip_bom(file_.get());
  if (plan.format == "parquet") { parquet_.reset(new ParquetReader(file_.get())); check_unchanged(); }
}
Reader::~Reader() = default;
void Reader::fail(const std::string &message) const
{
  throw FileSqlError("file SQL: " + plan_.path + ": record starting at line " + std::to_string(record_line_) + ": " + message);
}
std::string Reader::stamp() const
{
#ifdef _WIN32
  BY_HANDLE_FILE_INFORMATION info;
  if (!GetFileInformationByHandle(reinterpret_cast<HANDLE>(_get_osfhandle(_fileno(file_.get()))), &info)) fail("cannot stat opened file");
  return std::to_string(info.dwVolumeSerialNumber) + ":" + std::to_string(info.nFileIndexHigh) + ":"
    + std::to_string(info.nFileIndexLow) + ":" + std::to_string(info.nFileSizeHigh) + ":" + std::to_string(info.nFileSizeLow)
    + ":" + std::to_string(info.ftLastWriteTime.dwHighDateTime) + ":" + std::to_string(info.ftLastWriteTime.dwLowDateTime);
#else
  struct stat info;
  if (fstat(fileno(file_.get()), &info) != 0 || !S_ISREG(info.st_mode)) fail("cannot stat regular input");
  return std::to_string(info.st_dev) + ":" + std::to_string(info.st_ino) + ":" + std::to_string(info.st_size)
#if defined(__APPLE__)
    + ":" + std::to_string(info.st_mtimespec.tv_sec) + ":" + std::to_string(info.st_mtimespec.tv_nsec)
    + ":" + std::to_string(info.st_ctimespec.tv_sec) + ":" + std::to_string(info.st_ctimespec.tv_nsec);
#else
    + ":" + std::to_string(info.st_mtim.tv_sec) + ":" + std::to_string(info.st_mtim.tv_nsec)
    + ":" + std::to_string(info.st_ctim.tv_sec) + ":" + std::to_string(info.st_ctim.tv_nsec);
#endif
#endif
}
void Reader::check_unchanged() const
{
  if (check_) check_();
  if (plan_.format == "glob") return;
  if (stamp() != fingerprint_) fail("input changed during scan");
  if (checked_path(plan_.path, secure_root_) != plan_.path) fail("input path changed during scan");
#ifdef _WIN32
  HANDLE current = CreateFileW(from_utf8(plan_.path).c_str(), FILE_READ_ATTRIBUTES,
      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING, 0, nullptr);
  if (current == INVALID_HANDLE_VALUE) fail("cannot verify input path identity");
  BY_HANDLE_FILE_INFORMATION by_path, by_handle;
  bool ok = GetFileInformationByHandle(current, &by_path)
    && GetFileInformationByHandle(reinterpret_cast<HANDLE>(_get_osfhandle(_fileno(file_.get()))), &by_handle);
  CloseHandle(current);
  if (!ok || by_path.dwVolumeSerialNumber != by_handle.dwVolumeSerialNumber
      || by_path.nFileIndexHigh != by_handle.nFileIndexHigh || by_path.nFileIndexLow != by_handle.nFileIndexLow)
    fail("input was replaced during scan");
#else
  struct stat by_path, by_handle;
  if (::stat(plan_.path.c_str(), &by_path) != 0 || fstat(fileno(file_.get()), &by_handle) != 0
      || by_path.st_dev != by_handle.st_dev || by_path.st_ino != by_handle.st_ino) fail("input was replaced during scan");
#endif
}
int Reader::get()
{
  int value = fgetc(file_.get());
  if (value == '\n') ++line_;
  if (value == EOF && ferror(file_.get())) fail("input read failed");
  return value;
}
bool Reader::csv(Row &row)
{
  row.clear();
  std::string value;
  bool quoted = false, in_quotes = false, after_quote = false, any = false;
  size_t bytes = 0;
  const Options &options = plan_.options;
  for (;;) {
    int ch = get();
    if (++bytes > MAX_RECORD) fail("CSV record exceeds 16 MiB");
    if (ch == EOF) {
      if (in_quotes) fail("unterminated CSV quoted field");
      if (!any && row.empty() && value.empty()) return false;
    } else if (in_quotes) {
      if (ch == options.escape && options.escape != options.quote) {
        int next = get();
        if (next == EOF) fail("unterminated CSV escape");
        ++bytes;
        if (next != options.quote && next != options.escape) value += static_cast<char>(ch);
        value += static_cast<char>(next);
      } else if (ch == options.quote) {
        int next = get();
        if (next == options.quote && options.escape == options.quote) { ++bytes; value += options.quote; }
        else { if (next != EOF) { ungetc(next, file_.get()); if (next == '\n') --line_; } in_quotes = false; after_quote = true; }
      } else { value += static_cast<char>(ch); }
      any = true;
      continue;
    } else if (ch == options.quote && value.empty() && !after_quote) {
      quoted = true; in_quotes = true; any = true; continue;
    } else if (ch != options.delimiter && ch != '\r' && ch != '\n') {
      if (after_quote || ch == options.quote) fail("unexpected character in CSV field");
      value += static_cast<char>(ch); any = true; continue;
    }
    if (!valid_utf8(value)) fail("invalid UTF-8 in CSV field");
    if (!quoted && value == options.null_text) row.emplace_back(std::nullopt);
    else row.emplace_back(value);
    if (row.size() > MAX_COLUMNS) fail("CSV exceeds 1024 columns");
    value.clear(); quoted = false; after_quote = false;
    if (ch == EOF || ch == '\n' || ch == '\r') {
      if (ch == '\r') { int next = get(); if (next != '\n' && next != EOF) ungetc(next, file_.get()); }
      if (!any && row.size() == 1 && !row[0]) { row.clear(); if (ch == EOF) return false; bytes = 0; continue; }
      return true;
    }
    any = true;
  }
}
bool Reader::json(Row &row, std::vector<std::string> &keys)
{
  std::string line;
  for (;;) {
    int ch = get();
    if (ch == EOF && line.empty()) return false;
    if (ch == '\n' || ch == EOF) {
      if (!line.empty() && line.back() == '\r') line.pop_back();
      if (line.find_first_not_of(" \t\r") != std::string::npos) break;
      line.clear(); if (ch == EOF) return false;
    } else line += static_cast<char>(ch);
    if (line.size() > MAX_RECORD) fail("JSONL record exceeds 16 MiB");
  }
  if (first_record_ && line.compare(0, 3, "\xEF\xBB\xBF") == 0) line.erase(0, 3);
  first_record_ = false;
  rapidjson::Document document;
  document.Parse<rapidjson::kParseValidateEncodingFlag | rapidjson::kParseFullPrecisionFlag>(line.data(), line.size());
  if (document.HasParseError() || !document.IsObject()) fail("expected a flat JSON object");
  if (document.MemberCount() > MAX_COLUMNS) fail("JSONL exceeds 1024 fields");
  row.clear(); keys.clear(); json_types_.clear();
  std::set<std::string> seen;
  for (auto member = document.MemberBegin(); member != document.MemberEnd(); ++member) {
    std::string key(member->name.GetString(), member->name.GetStringLength());
    if (!seen.insert(key).second) fail("duplicate JSON object key: " + key);
    if (member->value.IsObject() || member->value.IsArray()) fail("nested JSON values are not supported: " + key);
    keys.push_back(key);
    Type type = Type::Null;
    if (member->value.IsString()) {
      type = infer_type(scalar(member->value));
      if (type != Type::Date && type != Type::Datetime) type = Type::Varchar;
    } else if (member->value.IsBool()) type = Type::Boolean;
    else if (member->value.IsInt64()) type = Type::Bigint;
    else if (member->value.IsUint64()) type = Type::Varchar;
    else if (member->value.IsDouble()) type = Type::Double;
    json_types_.push_back(type);
    if (member->value.IsNull()) row.emplace_back(std::nullopt);
    else row.emplace_back(scalar(member->value));
  }
  return true;
}
bool Reader::raw(Row &row, std::vector<std::string> &keys)
{
  // Bound change-detection and cancellation latency to one scan batch.
  if ((records_checked_++ % 64) == 0) check_unchanged();
  record_line_ = line_;
  bool found;
  if (plan_.format == "csv") {
    found = csv(row);
    if (found && first_record_) {
      if (row[0] && row[0]->compare(0, 3, "\xEF\xBB\xBF") == 0) row[0]->erase(0, 3);
      first_record_ = false;
    }
  } else if (plan_.format == "json") found = json(row, keys);
  else { fail("unsupported file format: " + plan_.format); found = false; }
  if (!found || (records_checked_ % 64) == 0) check_unchanged();
  return found;
}
bool Reader::next(Row &row)
{
  if (parquet_) {
    if ((records_checked_++ % 64) == 0) check_unchanged();
    bool found = parquet_->next(row);
    if (!found || (records_checked_ % 64) == 0) check_unchanged();
    return found;
  }
  if (plan_.format == "glob") {
    if (glob_index_ == glob_files_.size()) return false;
    row = {glob_files_[glob_index_++]}; return true;
  }
  Row values;
  std::vector<std::string> keys;
  if (plan_.format == "csv" && header_pending_) { raw(values, keys); header_pending_ = false; }
  if (!raw(values, keys)) return false;
  if (plan_.format == "csv") {
    if (values.size() != plan_.columns.size()) fail("CSV field count differs from schema");
    row = std::move(values);
  } else {
    row.assign(plan_.columns.size(), std::nullopt);
    for (size_t i = 0; i < keys.size(); ++i) {
      auto col = std::find_if(plan_.columns.begin(), plan_.columns.end(), [&](const Column &c) { return c.key == keys[i]; });
      if (col == plan_.columns.end()) fail("JSON field appeared after inference: " + keys[i]);
      row[static_cast<size_t>(col - plan_.columns.begin())] = std::move(values[i]);
    }
  }
  return true;
}
void Reader::project(const std::vector<size_t> &columns)
{
  if (parquet_) parquet_->project(columns);
}
void Reader::rescan()
{
  check_unchanged();
  records_checked_ = 0;
  if (parquet_) { parquet_->rescan(); return; }
  if (file_ && fseek(file_.get(), 0, SEEK_SET) != 0) fail("cannot rewind input");
  if (file_ && (plan_.format == "csv" || plan_.format == "json")) skip_bom(file_.get());
  line_ = 1; first_record_ = true; header_pending_ = plan_.options.header; glob_index_ = 0;
}
Plan infer(Plan source, const std::string &secure_root, std::function<void()> check)
{
  if (source.format == "glob") {
    source.rows = discover(source.path, secure_root).size();
    source.columns = {{"file", "file", Type::Varchar, false}}; return source;
  }
  source.columns.clear(); source.rows = 0; source.fingerprint.clear();
  Reader reader(source, secure_root, std::move(check));
  source.path = reader.plan_.path;
  source.fingerprint = reader.fingerprint_;
  if (reader.parquet_) {
    source.columns = reader.parquet_->columns(); source.rows = reader.parquet_->rows();
    unique_names(source.columns); reader.check_unchanged(); return source;
  }
  Row row;
  std::vector<std::string> keys;
  if (source.format == "csv" && source.options.header) {
    if (!reader.raw(row, keys)) throw FileSqlError("file SQL: CSV has no header");
    for (const auto &cell : row) source.columns.push_back({cell.value_or(""), "", Type::Null, false});
    unique_names(source.columns);
  }
  while (reader.raw(row, keys)) {
    if (source.format == "csv") {
      if (source.columns.empty()) {
        for (size_t i = 0; i < row.size(); ++i) source.columns.push_back({"column" + std::to_string(i), "", Type::Null, false});
      }
      if (row.size() != source.columns.size()) reader.fail("CSV field count differs from header");
      for (size_t i = 0; i < row.size(); ++i) {
        source.columns[i].nullable |= !row[i];
        if (row[i]) {
          source.columns[i].type = widen(source.columns[i].type, infer_type(*row[i]));
          if (datetime(*row[i]) && row[i]->size() > 19) source.columns[i].scale = (std::max)(source.columns[i].scale, static_cast<int>(row[i]->size() - 20));
        }
      }
    } else {
      std::set<std::string> present(keys.begin(), keys.end());
      for (auto &column : source.columns) if (!present.count(column.key)) column.nullable = true;
      for (size_t i = 0; i < keys.size(); ++i) {
        auto column = std::find_if(source.columns.begin(), source.columns.end(), [&](const Column &c) { return c.key == keys[i]; });
        if (column == source.columns.end()) {
          if (source.columns.size() >= MAX_COLUMNS) reader.fail("JSONL union exceeds 1024 columns");
          source.columns.push_back({keys[i], keys[i], Type::Null, source.rows != 0}); column = source.columns.end() - 1;
        }
        column->nullable |= !row[i];
        if (row[i]) {
          column->type = widen(column->type, reader.json_types_[i]);
          if (datetime(*row[i]) && row[i]->size() > 19) column->scale = (std::max)(column->scale, static_cast<int>(row[i]->size() - 20));
        }
      }
    }
    ++source.rows;
  }
  if (source.columns.empty()) throw FileSqlError("file SQL: input has no columns");
  unique_names(source.columns);
  for (auto &column : source.columns) if (source.rows == 0 || column.type == Type::Null) column.nullable = true;
  reader.check_unchanged();
  return source;
}

std::string encode(const Plan &plan)
{
  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  writer.StartObject();
  auto text = [&](const char *key, const std::string &value) { writer.Key(key); writer.String(value.data(), static_cast<rapidjson::SizeType>(value.size())); };
  text("format", plan.format); text("path", plan.path); text("fingerprint", plan.fingerprint);
  writer.Key("header"); writer.Bool(plan.options.header);
  text("delimiter", std::string(1, plan.options.delimiter)); text("quote", std::string(1, plan.options.quote));
  text("escape", std::string(1, plan.options.escape)); text("null_text", plan.options.null_text);
  writer.Key("rows"); writer.Uint64(plan.rows); writer.Key("columns"); writer.StartArray();
  for (const auto &column : plan.columns) {
    writer.StartObject(); text("name", column.name); text("key", column.key);
    writer.Key("type"); writer.Int(static_cast<int>(column.type)); writer.Key("nullable"); writer.Bool(column.nullable); writer.Key("scale"); writer.Int(column.scale); writer.EndObject();
  }
  writer.EndArray(); writer.EndObject();
  return std::string(buffer.GetString(), buffer.GetSize());
}
Plan decode(const std::string &json)
{
  rapidjson::Document doc;
  doc.Parse(json.data(), json.size());
  if (doc.HasParseError() || !doc.IsObject()) throw FileSqlError("file SQL: invalid scan definition");
  auto text = [&](const char *key) -> std::string {
    if (!doc.HasMember(key) || !doc[key].IsString()) throw FileSqlError("file SQL: invalid scan definition field");
    return std::string(doc[key].GetString(), doc[key].GetStringLength());
  };
  Plan result;
  result.format = text("format"); result.path = text("path"); result.fingerprint = text("fingerprint");
  result.options.null_text = text("null_text");
  auto character = [&](const char *key) { std::string value = text(key); if (value.size() != 1) throw FileSqlError("file SQL: invalid separator"); return value[0]; };
  result.options.delimiter = character("delimiter"); result.options.quote = character("quote"); result.options.escape = character("escape");
  if (!doc.HasMember("header") || !doc["header"].IsBool() || !doc.HasMember("rows") || !doc["rows"].IsUint64()
      || !doc.HasMember("columns") || !doc["columns"].IsArray() || doc["columns"].Size() > MAX_COLUMNS)
    throw FileSqlError("file SQL: invalid scan schema");
  result.options.header = doc["header"].GetBool(); result.rows = doc["rows"].GetUint64();
  for (const auto &column : doc["columns"].GetArray()) {
    if (!column.IsObject() || !column.HasMember("name") || !column["name"].IsString()
        || !column.HasMember("key") || !column["key"].IsString() || !column.HasMember("type") || !column["type"].IsInt()
        || column["type"].GetInt() < 0 || column["type"].GetInt() > static_cast<int>(Type::Datetime)
        || !column.HasMember("nullable") || !column["nullable"].IsBool()) throw FileSqlError("file SQL: invalid scan column");
    result.columns.push_back({std::string(column["name"].GetString(), column["name"].GetStringLength()),
                              std::string(column["key"].GetString(), column["key"].GetStringLength()),
                              static_cast<Type>(column["type"].GetInt()), column["nullable"].GetBool()});
    if (column.HasMember("scale") && column["scale"].IsInt() && column["scale"].GetInt() >= 0 && column["scale"].GetInt() <= 6) result.columns.back().scale = column["scale"].GetInt();
  }
  return result;
}

}}} // namespace oceanbase::sql::file_sql
