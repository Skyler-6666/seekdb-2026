/* Copyright (c) 2026 OceanBase. Licensed under the Apache License, Version 2.0. */
#include "file_copy_writer.h"
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fcntl.h>
#ifdef _WIN32
#include <io.h>
#include <sys/stat.h>
#else
#include <unistd.h>
#endif

namespace oceanbase { namespace sql { namespace file_sql {
namespace {
namespace fs = std::filesystem;
fs::path copy_path(const std::string &value)
{
  return fs::path(std::u8string(reinterpret_cast<const char8_t *>(value.data()), value.size()));
}
std::string copy_utf8(const fs::path &value)
{
  auto bytes = value.generic_u8string();
  return std::string(reinterpret_cast<const char *>(bytes.data()), bytes.size());
}
}
CsvCopyWriter::CsvCopyWriter(const Plan &plan, const std::string &secure_root)
  : plan_(plan), root_(secure_root)
{
  if (plan.path.find('\0') != std::string::npos) throw FileSqlError("file SQL: NUL in COPY path");
  fs::path path = copy_path(plan.path);
  if (path.filename().empty() || path.filename() == "." || path.filename() == "..") throw FileSqlError("file SQL: invalid COPY filename");
  parent_ = checked_path(path.has_parent_path() ? copy_utf8(path.parent_path()) : ".", root_);
  target_ = copy_utf8(copy_path(parent_) / path.filename());
  if (fs::exists(fs::symlink_status(copy_path(target_)))) throw FileSqlError("file SQL: COPY target already exists");
  static std::atomic<uint64_t> sequence{0};
  temporary_ = target_ + ".tmp-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())
    + "-" + std::to_string(sequence.fetch_add(1));
#ifdef _WIN32
  int descriptor = _wopen(copy_path(temporary_).c_str(), _O_WRONLY | _O_CREAT | _O_EXCL | _O_BINARY, _S_IREAD | _S_IWRITE);
  if (descriptor >= 0) { file_ = _fdopen(descriptor, "wb"); if (file_ == nullptr) _close(descriptor); }
#else
  int descriptor = ::open(temporary_.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
  if (descriptor >= 0) { file_ = fdopen(descriptor, "wb"); if (file_ == nullptr) ::close(descriptor); }
#endif
  if (file_ == nullptr) {
    if (descriptor >= 0) { std::error_code ignored; fs::remove(copy_path(temporary_), ignored); }
    throw FileSqlError("file SQL: cannot create COPY temporary file");
  }
  try {
    checked_path(temporary_, root_);
    if (plan.options.header) {
      Row header;
      for (const auto &column : plan.columns) header.emplace_back(column.name);
      write(header);
    }
  } catch (...) {
    fclose(file_); file_ = nullptr;
    std::error_code ignored; fs::remove(copy_path(temporary_), ignored);
    throw;
  }
}
CsvCopyWriter::~CsvCopyWriter()
{
  if (file_ != nullptr) fclose(file_);
  if (!temporary_.empty()) { std::error_code ignored; fs::remove(copy_path(temporary_), ignored); }
}
void CsvCopyWriter::write(const Row &row)
{
  if (file_ == nullptr || row.size() != plan_.columns.size()) throw FileSqlError("file SQL: invalid COPY row");
  std::string record;
  for (size_t i = 0; i < row.size(); ++i) {
    if (i != 0) record += plan_.options.delimiter;
    if (!row[i]) record += plan_.options.null_text;
    else {
      record += plan_.options.quote;
      for (char ch : *row[i]) {
        if (ch == plan_.options.quote || (ch == plan_.options.escape && plan_.options.escape != plan_.options.quote)) record += plan_.options.escape;
        record += ch;
        if (record.size() >= 16 * 1024 * 1024) throw FileSqlError("file SQL: COPY record exceeds 16 MiB");
      }
      record += plan_.options.quote;
    }
    if (record.size() >= 16 * 1024 * 1024) throw FileSqlError("file SQL: COPY record exceeds 16 MiB");
  }
  record += '\n';
  if (fwrite(record.data(), 1, record.size(), file_) != record.size()) throw FileSqlError("file SQL: COPY write failed");
}
void CsvCopyWriter::commit()
{
  if (file_ == nullptr) throw FileSqlError("file SQL: COPY writer already closed");
  if (fflush(file_) != 0) throw FileSqlError("file SQL: COPY flush failed");
  int result = fclose(file_); file_ = nullptr;
  if (result != 0) throw FileSqlError("file SQL: COPY close failed");
  if (checked_path(parent_, root_) != parent_ || checked_path(temporary_, root_) != temporary_) throw FileSqlError("file SQL: COPY path changed");
  std::error_code error;
  // Hard-link publication is atomic and fails if the target already exists.
  fs::create_hard_link(copy_path(temporary_), copy_path(target_), error);
  if (error) throw FileSqlError("file SQL: cannot publish COPY target: " + error.message());
  fs::remove(copy_path(temporary_), error);
  if (!error) temporary_.clear();
}
}}}
