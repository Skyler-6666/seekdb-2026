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

#include "sql/table_function/file_sql_scan.h"
#include "sql/table_function/file_sql_utils.h"
#include "sql/engine/ob_exec_context.h"
#include "common/object/ob_object.h"
#include "lib/allocator/page_arena.h"
#include "lib/container/ob_se_array.h"
#include <new>
#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>

namespace oceanbase
{
namespace sql
{
using namespace common;

namespace
{

const int64_t MAX_FILE_SQL_STRING_LENGTH = 1048576;

// Options string layout: "type=<n>;header=<0|1>;size=<n>;mtime=<n>;cols=<k1k2...>"
struct ScanOptions
{
  file_sql::FileType type_ = file_sql::FILE_TYPE_INVALID;
  bool header_ = true;
  int64_t size_ = -1;
  int64_t mtime_ = -1;
  file_sql::ColumnKind kinds_[64];
  int64_t kind_count_ = 0;
};

file_sql::ColumnKind char_to_kind(char c)
{
  switch (c) {
    case 'i': return file_sql::COLUMN_KIND_BIGINT;
    case 'd': return file_sql::COLUMN_KIND_DOUBLE;
    case 'b': return file_sql::COLUMN_KIND_BOOLEAN;
    case 'a': return file_sql::COLUMN_KIND_DATE;
    case 't': return file_sql::COLUMN_KIND_DATETIME;
    case 's': return file_sql::COLUMN_KIND_VARCHAR;
    default: return file_sql::COLUMN_KIND_NULL;
  }
}

int parse_scan_options(const ObString &text, ScanOptions &options)
{
  int ret = OB_SUCCESS;
  if (0 == text.length()) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    std::string serialized(text.ptr(), static_cast<size_t>(text.length()));
    size_t pos = 0;
    int64_t type = file_sql::FILE_TYPE_INVALID;
    while (OB_SUCC(ret) && pos < serialized.size()) {
      const size_t semicolon = serialized.find(';', pos);
      const std::string item = serialized.substr(pos, std::string::npos == semicolon ?
          std::string::npos : semicolon - pos);
      pos = (std::string::npos == semicolon) ? serialized.size() : semicolon + 1;
      const size_t equal = item.find('=');
      if (std::string::npos == equal) { continue; }
      const std::string key = item.substr(0, equal);
      const std::string value = item.substr(equal + 1);
      int64_t parsed = 0;
      if ("type" == key) {
        if (file_sql::parse_int64(value, parsed)) { type = parsed; }
      } else if ("header" == key) {
        if (file_sql::parse_int64(value, parsed)) { options.header_ = parsed != 0; }
      } else if ("size" == key) {
        if (file_sql::parse_int64(value, parsed)) { options.size_ = parsed; }
      } else if ("mtime" == key) {
        if (file_sql::parse_int64(value, parsed)) { options.mtime_ = parsed; }
      } else if ("cols" == key) {
        for (size_t i = 0; i < value.size(); ++i) {
          if (options.kind_count_ >= 64) {
            ret = OB_NOT_SUPPORTED;
          } else {
            options.kinds_[options.kind_count_++] = char_to_kind(value[i]);
          }
        }
      }
    }
    if (OB_SUCC(ret)) {
      options.type_ = static_cast<file_sql::FileType>(type);
      if (options.kind_count_ <= 0 || file_sql::FILE_TYPE_INVALID == options.type_) {
        ret = OB_INVALID_ARGUMENT;
      }
    }
  }
  return ret;
}

int build_cell(const file_sql::ColumnKind kind, const file_sql::CellValue &value,
               ObIAllocator &allocator, ObObj &cell)
{
  int ret = OB_SUCCESS;
  if (value.is_null_) {
    cell.set_null();
  } else if (file_sql::COLUMN_KIND_BIGINT == kind) {
    int64_t parsed = 0;
    if (!file_sql::parse_int64(value.text_, parsed)) {
      ret = OB_INVALID_DATA;
    } else {
      cell.set_int(parsed);
    }
  } else if (file_sql::COLUMN_KIND_DOUBLE == kind) {
    double parsed = 0.0;
    if (!file_sql::parse_double(value.text_, parsed)) {
      ret = OB_INVALID_DATA;
    } else {
      cell.set_double(parsed);
    }
  } else if (file_sql::COLUMN_KIND_BOOLEAN == kind) {
    bool parsed = false;
    int64_t number = 0;
    if (file_sql::parse_boolean(value.text_, parsed)) {
      cell.set_tinyint(static_cast<int8_t>(parsed ? 1 : 0));
    } else if (!file_sql::parse_int64(value.text_, number) || (0 != number && 1 != number)) {
      ret = OB_INVALID_DATA;
    } else {
      cell.set_tinyint(static_cast<int8_t>(number));
    }
  } else if (file_sql::COLUMN_KIND_DATE == kind) {
    int32_t days = 0;
    if (!file_sql::parse_date_days(value.text_, days)) {
      ret = OB_INVALID_DATA;
    } else {
      cell.set_date(days);
    }
  } else if (file_sql::COLUMN_KIND_DATETIME == kind) {
    int64_t micros = 0;
    if (!file_sql::parse_datetime_micros(value.text_, micros)) {
      ret = OB_INVALID_DATA;
    } else {
      cell.set_datetime(micros);
    }
  } else {
    const int64_t length = static_cast<int64_t>(value.text_.size());
    if (length > MAX_FILE_SQL_STRING_LENGTH) {
      ret = OB_SIZE_OVERFLOW;
    } else {
      char *buffer = static_cast<char *>(allocator.alloc(length + 1));
      if (NULL == buffer) {
        ret = OB_ALLOCATE_MEMORY_FAILED;
      } else {
        if (length > 0) { MEMCPY(buffer, value.text_.data(), length); }
        buffer[length] = '\0';
        cell.set_varchar(ObString(static_cast<int32_t>(length), buffer));
        cell.set_collation_type(CS_TYPE_UTF8MB4_BIN);
      }
    }
  }
  return ret;
}

class FileTableScan : public ITableFunctionScan
{
public:
  explicit FileTableScan(const TableFunctionScanContext &context)
      : projection_(context.column_ids_),
        arguments_(context.arguments_),
        reader_(NULL),
        row_allocator_("FileSqlRow"),
        opened_(false) {}

  int open() override
  {
    int ret = OB_SUCCESS;
    if (arguments_.count() < 2) {
      ret = OB_INVALID_ARGUMENT;
    } else if (OB_FAIL(parse_scan_options(arguments_.at(1).get_string(), options_))) {
    } else {
      path_ = arguments_.at(0).get_string();
      if (OB_FAIL(open_reader())) {
      } else {
        opened_ = true;
      }
    }
    return ret;
  }

  int rescan(const ObIArray<ObObj> &arguments) override
  {
    UNUSED(arguments);
    return open();
  }

  int get_next_row(ObNewRow *&row) override
  {
    int ret = OB_SUCCESS;
    file_sql::RowValues values;
    row_allocator_.reuse();
    if (!opened_ || NULL == reader_) {
      ret = OB_ERR_UNEXPECTED;
    } else {
      const int read_ret = reader_->next_row(values);
      if (1 == read_ret) {
        ret = OB_ITER_END;
      } else if (read_ret < 0) {
        LOG_WARN("failed to read file row");
        ret = OB_INVALID_DATA;
      } else if (static_cast<int64_t>(values.cells_.size()) != options_.kind_count_) {
        LOG_WARN("file row field count mismatch", K(values.line_no_),
                 K(values.cells_.size()), K(options_.kind_count_));
        ret = OB_INVALID_DATA;
      } else {
        cells_.reuse();
        for (int64_t i = 0; OB_SUCC(ret) && i < projection_.count(); ++i) {
          const int64_t index = static_cast<int64_t>(projection_.at(i)) - OB_APP_MIN_COLUMN_ID;
          ObObj cell;
          if (index < 0 || index >= options_.kind_count_) {
            ret = OB_ERR_UNEXPECTED;
          } else if (OB_FAIL(build_cell(options_.kinds_[index], values.cells_[index],
                                        row_allocator_, cell))) {
          } else if (OB_FAIL(cells_.push_back(cell))) {
          }
        }
        if (OB_SUCC(ret)) {
          row_.cells_ = cells_.get_data();
          row_.count_ = cells_.count();
          row = &row_;
        }
      }
    }
    return ret;
  }

  int close() override
  {
    if (NULL != reader_) {
      reader_->close();
      delete reader_;
      reader_ = NULL;
    }
    opened_ = false;
    row_allocator_.reuse();
    return OB_SUCCESS;
  }

private:
  int open_reader()
  {
    int ret = OB_SUCCESS;
    std::string path(path_.ptr(), static_cast<size_t>(path_.length()));
    file_sql::FileStamp stamp;
    if (OB_FAIL(file_sql::read_file_stamp(path, stamp))) {
      LOG_USER_ERROR(OB_INVALID_ARGUMENT, "file does not exist or is not a regular file");
      ret = OB_INVALID_ARGUMENT;
    } else if ((options_.size_ >= 0 && stamp.size_ != options_.size_) ||
               (options_.mtime_ >= 0 && stamp.mtime_ != options_.mtime_)) {
      LOG_USER_ERROR(OB_INVALID_ARGUMENT, "file changed after the statement was bound");
      ret = OB_INVALID_ARGUMENT;
    } else if (NULL == (reader_ = file_sql::create_row_reader(options_.type_))) {
      LOG_USER_ERROR(OB_NOT_SUPPORTED, "this file format");
      ret = OB_NOT_SUPPORTED;
    } else {
      file_sql::Options options;
      options.type_ = options_.type_;
      options.header_ = options_.header_;
      if (OB_FAIL(reader_->open(path, options))) {
        LOG_USER_ERROR(OB_INVALID_ARGUMENT, "failed to open file");
        ret = OB_INVALID_ARGUMENT;
      }
    }
    return ret;
  }

  const ObIArray<uint64_t> &projection_;
  const ObIArray<ObObj> &arguments_;
  ObString path_;
  ScanOptions options_;
  file_sql::RowReader *reader_;
  ObArenaAllocator row_allocator_;
  ObSEArray<ObObj, 16> cells_;
  ObNewRow row_;
  bool opened_;
};

class GlobTableScan : public ITableFunctionScan
{
public:
  explicit GlobTableScan(const TableFunctionScanContext &context)
      : allocator_(context.allocator_), arguments_(context.arguments_), position_(0) {}

  int open() override
  {
    int ret = OB_SUCCESS;
    files_.clear();
    position_ = 0;
    if (arguments_.count() < 1) {
      ret = OB_INVALID_ARGUMENT;
    } else {
      const ObString pattern = arguments_.at(0).get_string();
      std::string text(pattern.ptr(), static_cast<size_t>(pattern.length()));
      if (OB_FAIL(file_sql::expand_glob(text, files_))) {
        LOG_USER_ERROR(OB_INVALID_ARGUMENT, "invalid glob pattern");
        ret = OB_INVALID_ARGUMENT;
      }
    }
    return ret;
  }

  int rescan(const ObIArray<ObObj> &arguments) override
  {
    UNUSED(arguments);
    return open();
  }

  int get_next_row(ObNewRow *&row) override
  {
    int ret = OB_SUCCESS;
    if (position_ >= static_cast<int64_t>(files_.size())) {
      ret = OB_ITER_END;
    } else {
      const std::string &file = files_[position_++];
      char *buffer = static_cast<char *>(allocator_.alloc(static_cast<int64_t>(file.size()) + 1));
      if (NULL == buffer) {
        ret = OB_ALLOCATE_MEMORY_FAILED;
      } else {
        MEMCPY(buffer, file.data(), file.size());
        buffer[file.size()] = '\0';
        cell_.set_varchar(ObString(static_cast<int32_t>(file.size()), buffer));
        cell_.set_collation_type(CS_TYPE_UTF8MB4_BIN);
        row_.cells_ = &cell_;
        row_.count_ = 1;
        row = &row_;
      }
    }
    return ret;
  }

  int close() override
  {
    files_.clear();
    position_ = 0;
    return OB_SUCCESS;
  }

private:
  ObIAllocator &allocator_;
  const ObIArray<ObObj> &arguments_;
  std::vector<std::string> files_;
  int64_t position_;
  ObObj cell_;
  ObNewRow row_;
};

} // namespace

int create_file_table_scan(const TableFunctionScanContext &context, ITableFunctionScan *&scan)
{
  int ret = OB_SUCCESS;
  void *memory = context.allocator_.alloc(sizeof(FileTableScan));
  if (NULL == memory) {
    ret = OB_ALLOCATE_MEMORY_FAILED;
  } else {
    scan = new (memory) FileTableScan(context);
  }
  return ret;
}

int create_glob_table_scan(const TableFunctionScanContext &context, ITableFunctionScan *&scan)
{
  int ret = OB_SUCCESS;
  void *memory = context.allocator_.alloc(sizeof(GlobTableScan));
  if (NULL == memory) {
    ret = OB_ALLOCATE_MEMORY_FAILED;
  } else {
    scan = new (memory) GlobTableScan(context);
  }
  return ret;
}

} // namespace sql
} // namespace oceanbase
