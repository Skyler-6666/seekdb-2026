/* Copyright (c) 2026 OceanBase. Licensed under the Apache License, Version 2.0. */
#pragma once
#include "file_scan_utils.h"

namespace oceanbase { namespace sql { namespace file_sql {
// Footer metadata and one bounded page per projected column are retained.
class ParquetReader {
public:
  explicit ParquetReader(FILE *input);
  ~ParquetReader();
  const std::vector<Column> &columns() const;
  uint64_t rows() const;
  void project(const std::vector<size_t> &columns);
  bool next(Row &row);
  void rescan();
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}}}
