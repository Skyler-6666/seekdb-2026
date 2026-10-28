/* Copyright (c) 2026 OceanBase. Licensed under the Apache License, Version 2.0. */
#include "sql/engine/basic/file_scan_utils.h"
#include "sql/engine/basic/file_copy_writer.h"
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <chrono>
#ifndef FILE_SQL_STANDALONE_TEST
#include <gtest/gtest.h>
#endif

namespace {
using namespace oceanbase::sql::file_sql;
namespace fs = std::filesystem;
void check(bool condition)
{
  if (!condition) throw std::runtime_error("file scan regression assertion failed");
}
void rejects(const std::function<void()> &action)
{
  static int sequence = 0;
  ++sequence;
  try { action(); } catch (const FileSqlError &) { return; }
  throw std::runtime_error("expected file scan error, rejection case " + std::to_string(sequence));
}
void regressions()
{
  fs::path root = fs::current_path() / ("file-scan-tests-" + std::to_string(
      std::chrono::steady_clock::now().time_since_epoch().count()));
  check(fs::create_directory(root));
  const std::string allowed = root.generic_string();
  auto write = [&](const char *name, const std::string &data) {
    std::ofstream out(root / name, std::ios::binary); out << data; out.close(); check(!out.fail());
    return (root / name).generic_string();
  };
  Plan csv;
  csv.format = "csv";
  csv.path = write("basic.csv", "\xEF\xBB\xBF\"id\",id,score,flag,day,created,note,empty\r\n"
      "1,11,2.5,true,2024-02-29,2024-02-29T01:02:03,\"你好\nworld\",\r\n"
      "2,12,3,false,,2024-03-01 01:02:03,\"say \"\"hi\"\"\",\r\n");
  csv = infer(csv, allowed);
  check(csv.rows == 2 && csv.columns.size() == 8 && csv.columns[1].name == "id__2");
  check(csv.columns[0].type == Type::Bigint && csv.columns[2].type == Type::Double);
  check(csv.columns[3].type == Type::Boolean && csv.columns[4].type == Type::Date && csv.columns[4].nullable);
  check(csv.columns[5].type == Type::Datetime && csv.columns[7].type == Type::Null && csv.columns[7].nullable);
  Reader reader(csv, allowed);
  Row row;
  check(reader.next(row) && row[0] == Cell("1") && row[6] == Cell("你好\nworld") && !row[7]);
  check(reader.next(row) && row[6] == Cell("say \"hi\""));
  check(!reader.next(row)); reader.rescan(); check(reader.next(row) && row[0] == Cell("1"));
  Plan restored = decode(encode(csv));
  check(restored.fingerprint == csv.fingerprint && restored.columns[1].name == "id__2");
  check(infer_type("9223372036854775807") == Type::Bigint);
  check(infer_type("-9223372036854775808") == Type::Bigint);
  check(infer_type("9223372036854775808") == Type::Varchar);
  check(infer_type("1e309") == Type::Varchar && infer_type("NaN") == Type::Varchar);
  check(infer_type("2023-02-29") == Type::Varchar && infer_type("2024-02-29") == Type::Date);
  check(infer_type("2024-01-01T24:00:00") == Type::Varchar);
  check(widen(Type::Date, Type::Datetime) == Type::Datetime);
  check(widen(Type::Boolean, Type::Bigint) == Type::Varchar);

  Plan json;
  json.format = "json";
  json.path = write("union.jsonl", "{\"id\":1,\"v\":1,\"flag\":true}\n"
      "{\"id\":2,\"v\":1.5,\"extra\":\"世界\"}\n"
      "{\"id\":3,\"v\":\"3\",\"extra\":null}\n");
  json = infer(json, allowed);
  check(json.rows == 3 && json.columns.size() == 4 && json.columns[1].type == Type::Varchar);
  check(json.columns[2].type == Type::Boolean && json.columns[2].nullable && json.columns[3].nullable);
  Reader json_reader(json, allowed);
  check(json_reader.next(row) && row[1] == Cell("1") && !row[3]);
  check(json_reader.next(row) && !row[2] && row[3] == Cell("世界"));
  Plan bad = json;
  for (const char *data : {"{\"a\":{}}\n", "{\"a\":[]}\n", "{\"a\":1,\"a\":2}\n", "[1,2]\n", "{\"a\":\n"}) {
    bad.path = write("bad.jsonl", data); rejects([&] { infer(bad, allowed); });
  }
  bad = csv;
  for (const char *data : {"a,b\n1\n", "a,b\n\"unterminated\n", "a\n\xff\n"}) {
    bad.path = write("bad.csv", data); rejects([&] { infer(bad, allowed); });
  }
  Plan changed;
  changed.format = "csv"; changed.path = write("changed.csv", "id\n1\n"); changed = infer(changed, allowed);
  Reader changed_reader(changed, allowed);
  { std::ofstream out(root / "changed.csv", std::ios::app); out << "2\n"; }
  rejects([&] { changed_reader.next(row); });
  rejects([&] { Reader again(changed, allowed); });
  rejects([&] { checked_path((root / "../").generic_string(), allowed); });
  rejects([&] { checked_path(csv.path, ""); });
  rejects([&] { checked_path(csv.path, "NULL"); });
  auto files = discover((root / "*.jsonl").generic_string(), allowed);
  check(files.size() == 2 && files[0] < files[1]);
  // Decode the published fixture so this test needs no Arrow dependency.
  fs::path suite = fs::path(__FILE__).parent_path() / "../../../../tools/deploy/mysql_test/test_suite/file_sql/data/simple.parquet.b64";
  std::ifstream encoded_fixture(suite, std::ios::binary);
  check(encoded_fixture.is_open());
  std::string alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string parquet_bytes;
  uint32_t packed = 0; int bits = 0; char character;
  while (encoded_fixture.get(character)) {
    size_t digit = alphabet.find(character);
    if (digit == std::string::npos) continue;
    packed = (packed << 6) | static_cast<uint32_t>(digit); bits += 6;
    if (bits >= 8) { bits -= 8; parquet_bytes += static_cast<char>(packed >> bits); }
  }
  Plan parquet;
  parquet.format = "parquet"; parquet.path = write("simple.parquet", parquet_bytes);
  parquet = infer(parquet, allowed);
  check(parquet.rows == 7 && parquet.columns.size() == 1 && parquet.columns[0].type == Type::Bigint);
  Reader parquet_reader(parquet, allowed);
  for (int i = 0; i < 7; ++i) check(parquet_reader.next(row) && row[0] == Cell(std::to_string(i)));
  check(!parquet_reader.next(row)); parquet_reader.rescan(); check(parquet_reader.next(row) && row[0] == Cell("0"));
  Reader count_only(parquet, allowed); count_only.project({});
  for (int i = 0; i < 7; ++i) check(count_only.next(row) && !row[0]);
  Plan malformed; malformed.format = "parquet";
  malformed.path = write("truncated.parquet", parquet_bytes.substr(0, parquet_bytes.size() - 1));
  rejects([&] { infer(malformed, allowed); });
  malformed.path = write("bad-magic.parquet", std::string("FAIL") + parquet_bytes.substr(4));
  rejects([&] { infer(malformed, allowed); });
  std::cout << "PASS: published Parquet schema/rows, projection, rescan and corrupt/truncated input rejection\n";

  Plan copy;
  copy.format = "csv"; copy.path = (root / "copy.csv").generic_string();
  copy.options.delimiter = ';'; copy.options.null_text = "NULL";
  copy.columns = {{"text", "", Type::Varchar, true}, {"empty", "", Type::Varchar, true}, {"missing", "", Type::Varchar, true}};
  Row original = {"你好;\n\"quote\"", "", std::nullopt};
  { CsvCopyWriter writer(copy, allowed); writer.write(original); writer.commit(); }
  Plan copied = infer(copy, allowed); Reader exported(copied, allowed);
  check(exported.next(row) && row == original && !exported.next(row));
  rejects([&] { CsvCopyWriter duplicate(copy, allowed); });
  copy.path = (root / "aborted.csv").generic_string();
  { CsvCopyWriter writer(copy, allowed); writer.write(original); }
  check(!fs::exists(copy.path));
  for (auto &entry : fs::directory_iterator(root)) check(entry.path().filename().string().find(".tmp-") == std::string::npos);
  copy.path = (root / "../escape.csv").generic_string();
  rejects([&] { CsvCopyWriter outside(copy, allowed); });
  Plan fraction; fraction.format = "csv";
  fraction.path = write("fraction.csv", "created\n2024-02-29 01:02:03.123456\n2024-02-29 01:02:03\n");
  fraction = infer(fraction, allowed);
  check(fraction.columns[0].scale == 6 && decode(encode(fraction)).columns[0].scale == 6);
  Plan replaced; replaced.format = "csv"; replaced.path = write("replace.csv", "id\n1\n");
  replaced = infer(replaced, allowed); Reader old(replaced, allowed);
  fs::rename(root / "replace.csv", root / "previous.csv");
  write("replace.csv", "id\n2\n");
  rejects([&] { old.next(row); });
  std::cout << "PASS: COPY escaping/NULL roundtrip, no overwrite, aborted-file cleanup, time precision and path replacement\n";
  std::cout << "PASS: CSV/JSONL streaming, rescan, types, schema union, malformed records, file changes, paths and glob\n";
}
}

#ifdef FILE_SQL_STANDALONE_TEST
int main()
{
  try { regressions(); return 0; }
  catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
#else
TEST(FileScanUtils, streaming_and_validation) { EXPECT_NO_THROW(regressions()); }
#endif
