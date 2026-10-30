/* Copyright (c) 2026 OceanBase. Licensed under the Apache License, Version 2.0. */
#pragma once
#include "file_scan_utils.h"

namespace oceanbase { namespace sql { namespace file_sql {
class CsvCopyWriter {
public:
  CsvCopyWriter(const Plan &plan, const std::string &secure_root);
  ~CsvCopyWriter();
  CsvCopyWriter(const CsvCopyWriter &) = delete;
  CsvCopyWriter &operator=(const CsvCopyWriter &) = delete;
  void write(const Row &row);
  void commit();
private:
  Plan plan_;
  std::string root_, parent_, temporary_, target_;
  FILE *file_ = nullptr;
};
}}}
