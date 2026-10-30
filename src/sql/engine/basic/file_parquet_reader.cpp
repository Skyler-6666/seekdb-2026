/* Copyright (c) 2026 OceanBase. Licensed under the Apache License, Version 2.0. */
// Format reference: https://github.com/apache/parquet-format/blob/master/src/main/thrift/parquet.thrift
#include "file_parquet_reader.h"
#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>

namespace oceanbase { namespace sql { namespace file_sql {
namespace {
constexpr size_t MAX_PAGE = 16 * 1024 * 1024;
constexpr size_t MAX_PAGE_VALUES = 262144;
constexpr size_t MAX_MEMORY = 256 * 1024 * 1024;
[[noreturn]] void bad(const std::string &message) { throw FileSqlError("file SQL: Parquet: " + message); }
struct Bytes {
  const unsigned char *data = nullptr;
  size_t size = 0, pos = 0;
  explicit Bytes(const std::string &s) : data(reinterpret_cast<const unsigned char *>(s.data())), size(s.size()) {}
  Bytes(const unsigned char *p, size_t n) : data(p), size(n) {}
  uint8_t byte() { if (pos >= size) bad("truncated input"); return data[pos++]; }
  uint64_t varint() {
    uint64_t value = 0;
    for (unsigned shift = 0; shift < 64; shift += 7) {
      uint8_t b = byte();
      if (shift == 63 && (b & 0xfe)) bad("integer overflow");
      value |= static_cast<uint64_t>(b & 0x7f) << shift;
      if (!(b & 0x80)) return value;
    }
    bad("invalid varint");
  }
  int64_t zigzag() { uint64_t v = varint(); return static_cast<int64_t>((v >> 1) ^ (0 - (v & 1))); }
  uint64_t little(size_t n) {
    if (n > 8 || n > size - pos) bad("truncated fixed-width value");
    uint64_t value = 0;
    for (size_t i = 0; i < n; ++i) value |= static_cast<uint64_t>(byte()) << (8 * i);
    return value;
  }
  std::string string(size_t n) {
    if (n > size - pos || n > MAX_PAGE) bad("invalid string length");
    std::string value(reinterpret_cast<const char *>(data + pos), n); pos += n; return value;
  }
};
struct Node {
  int64_t number = 0;
  std::string text;
  std::map<int, Node> fields;
  std::vector<Node> items;
  bool has(int id) const { return fields.find(id) != fields.end(); }
  const Node &at(int id) const { auto found = fields.find(id); if (found == fields.end()) bad("required metadata field missing: " + std::to_string(id)); return found->second; }
  int64_t num(int id, int64_t fallback) const { return has(id) ? at(id).number : fallback; }
};
Node compact(Bytes &input, int type, size_t &nodes, unsigned depth = 0, bool field = false)
{
  if (++nodes > 200000 || depth > 32) bad("metadata resource limit exceeded");
  Node value;
  if (type == 1 || type == 2) value.number = field ? type == 1 : input.byte() == 1;
  else if (type == 3) value.number = static_cast<int8_t>(input.byte());
  else if (type >= 4 && type <= 6) value.number = input.zigzag();
  else if (type == 7) { input.little(8); }
  else if (type == 8) value.text = input.string(input.varint());
  else if (type == 9 || type == 10) {
    uint8_t header = input.byte(); size_t count = header >> 4; int element = header & 15;
    if (count == 15) count = input.varint();
    if (count > 100000 || count > input.size - input.pos) bad("invalid metadata list length");
    for (size_t i = 0; i < count; ++i) value.items.push_back(compact(input, element, nodes, depth + 1));
  } else if (type == 11) {
    size_t count = input.varint();
    if (count > 100000) bad("invalid metadata map length");
    if (count != 0) {
      uint8_t types = input.byte();
      for (size_t i = 0; i < count; ++i) {
        compact(input, types >> 4, nodes, depth + 1); compact(input, types & 15, nodes, depth + 1);
      }
    }
  } else if (type == 12) {
    int id = 0;
    for (;;) {
      uint8_t header = input.byte();
      if (header == 0) break;
      id = header >> 4 ? id + (header >> 4) : static_cast<int>(input.zigzag());
      if (id <= 0 || id > 32767 || value.has(id)) bad("invalid metadata field id");
      value.fields.emplace(id, compact(input, header & 15, nodes, depth + 1, true));
    }
  } else bad("unsupported compact-protocol type");
  return value;
}
std::string snappy(const std::string &compressed, size_t expected)
{
  Bytes in(compressed);
  if (expected > MAX_PAGE || in.varint() != expected) bad("invalid Snappy output length");
  std::string output; output.reserve(expected);
  while (in.pos < in.size) {
    uint8_t tag = in.byte(); size_t length = 0, offset = 0;
    if ((tag & 3) == 0) {
      length = tag >> 2;
      if (length < 60) ++length;
      else length = static_cast<size_t>(in.little(length - 59)) + 1;
      if (length > expected - output.size()) bad("Snappy literal exceeds output");
      output += in.string(length);
    } else {
      if ((tag & 3) == 1) { length = 4 + ((tag >> 2) & 7); offset = ((tag & 0xe0) << 3) | in.byte(); }
      else { length = 1 + (tag >> 2); offset = in.little((tag & 3) == 2 ? 2 : 4); }
      if (offset == 0 || offset > output.size() || length > expected - output.size()) bad("invalid Snappy back-reference");
      for (size_t i = 0; i < length; ++i) output += output[output.size() - offset];
    }
  }
  if (output.size() != expected) bad("truncated Snappy output");
  return output;
}
std::string unpack(const std::string &compressed, size_t expected, int codec)
{
  if (expected > MAX_PAGE) bad("page exceeds 16 MiB");
  if (codec == 1) return snappy(compressed, expected);
  if (codec != 0) bad("unsupported compression codec: " + std::to_string(codec));
  if (compressed.size() != expected) bad("uncompressed page size mismatch");
  return compressed;
}
uint32_t crc32(const std::string &data)
{
  uint32_t crc = 0xffffffff;
  for (unsigned char byte : data) {
    crc ^= byte;
    for (int i = 0; i < 8; ++i) crc = (crc >> 1) ^ (0xedb88320u & (0u - (crc & 1)));
  }
  return ~crc;
}
std::vector<uint32_t> hybrid(Bytes &in, unsigned width, size_t count)
{
  if (width > 32 || count > MAX_PAGE_VALUES) bad("invalid RLE width or count");
  std::vector<uint32_t> result; result.reserve(count);
  while (result.size() < count) {
    uint64_t header = in.varint();
    if (header <= 1) bad("empty RLE run");
    uint64_t run = header >> 1;
    if (header & 1) {
      if (run > (MAX_PAGE / std::max(1u, width))) bad("oversized bit-packed run");
      size_t values = static_cast<size_t>(run) * 8;
      size_t bytes = static_cast<size_t>(run) * width;
      if (bytes > in.size - in.pos) bad("truncated bit-packed run");
      size_t start = in.pos; in.pos += bytes;
      for (size_t i = 0; i < values && result.size() < count; ++i) {
        uint32_t value = 0;
        for (unsigned b = 0; b < width; ++b) {
          size_t bit = i * width + b;
          value |= static_cast<uint32_t>((in.data[start + bit / 8] >> (bit % 8)) & 1) << b;
        }
        result.push_back(value);
      }
    } else {
      uint64_t value = in.little((width + 7) / 8);
      if (width < 32 && value >= (1ULL << width)) bad("RLE value exceeds bit width");
      size_t needed = count - result.size();
      if (run > needed) bad("RLE run exceeds page value count");
      result.insert(result.end(), static_cast<size_t>(run), static_cast<uint32_t>(value));
    }
  }
  return result;
}
std::string number_text(double value)
{
  if (!std::isfinite(value)) bad("non-finite floating-point value");
  char buffer[128];
  auto result = std::to_chars(buffer, buffer + sizeof(buffer), value, std::chars_format::general, std::numeric_limits<double>::max_digits10);
  if (result.ec != std::errc()) bad("floating-point conversion failed");
  return std::string(buffer, result.ptr);
}
std::string date_text(int64_t days)
{
  if (days < -719162 || days > 2932896) bad("date outside SQL range");
  std::chrono::year_month_day date{std::chrono::sys_days{std::chrono::days{days}}};
  char buffer[32];
  snprintf(buffer, sizeof(buffer), "%04d-%02u-%02u", static_cast<int>(date.year()), static_cast<unsigned>(date.month()), static_cast<unsigned>(date.day()));
  return buffer;
}
std::string timestamp_text(int64_t micros)
{
  constexpr int64_t DAY = 86400000000LL;
  int64_t days = micros / DAY, time = micros % DAY;
  if (time < 0) { --days; time += DAY; }
  std::string result = date_text(days);
  char buffer[48];
  snprintf(buffer, sizeof(buffer), " %02lld:%02lld:%02lld.%06lld", static_cast<long long>(time / 3600000000LL),
      static_cast<long long>((time / 60000000LL) % 60), static_cast<long long>((time / 1000000LL) % 60), static_cast<long long>(time % 1000000LL));
  return result + buffer;
}
struct Field {
  Column column;
  int physical = -1, converted = -1, length = 0, timestamp_unit = 0;
};
std::string plain(Bytes &in, const Field &field, size_t index)
{
  if (field.physical == 0) { if (index / 8 >= in.size - in.pos) bad("truncated boolean values"); return (in.data[in.pos + index / 8] >> (index % 8)) & 1 ? "true" : "false"; }
  if (field.physical == 1) {
    uint32_t bits = static_cast<uint32_t>(in.little(4));
    if (field.column.type == Type::Date) return date_text(static_cast<int32_t>(bits));
    return field.converted >= 11 && field.converted <= 13 ? std::to_string(bits) : std::to_string(static_cast<int32_t>(bits));
  }
  if (field.physical == 2) {
    uint64_t bits = in.little(8);
    if (field.column.type == Type::Datetime) {
      int64_t value = static_cast<int64_t>(bits);
      if (field.timestamp_unit == 1) {
        if (value < std::numeric_limits<int64_t>::min() / 1000 || value > std::numeric_limits<int64_t>::max() / 1000) bad("timestamp overflow");
        value *= 1000;
      } else if (field.timestamp_unit == 3) value /= 1000;
      return timestamp_text(value);
    }
    return field.converted == 14 ? std::to_string(bits) : std::to_string(static_cast<int64_t>(bits));
  }
  if (field.physical == 3) {
    uint64_t nanos = in.little(8); int64_t days = static_cast<int64_t>(in.little(4)) - 2440588;
    if (nanos >= 86400000000000ULL || days < -719162 || days > 2932896) bad("invalid INT96 timestamp");
    return timestamp_text(days * 86400000000LL + static_cast<int64_t>(nanos / 1000));
  }
  if (field.physical == 4) { uint32_t bits = static_cast<uint32_t>(in.little(4)); float value; memcpy(&value, &bits, 4); return number_text(value); }
  if (field.physical == 5) { uint64_t bits = in.little(8); double value; memcpy(&value, &bits, 8); return number_text(value); }
  if (field.physical == 6) { size_t size = in.little(4); return in.string(size); }
  if (field.physical == 7) return in.string(field.length);
  bad("unsupported physical type");
}
std::vector<int64_t> delta_values(Bytes &input, size_t count)
{
  uint64_t block = input.varint(), miniblocks = input.varint(), total = input.varint();
  if (block == 0 || block > MAX_PAGE_VALUES || block % 128 != 0 || miniblocks == 0
      || miniblocks > block || block % miniblocks != 0 || (block / miniblocks) % 32 != 0
      || total != count || count > MAX_PAGE_VALUES) bad("invalid delta header");
  std::vector<int64_t> result;
  if (count == 0) return result;
  result.reserve(count); result.push_back(input.zigzag());
  while (result.size() < count) {
    int64_t minimum = input.zigzag();
    std::vector<unsigned> widths;
    for (size_t i = 0; i < miniblocks; ++i) widths.push_back(input.byte());
    for (unsigned width : widths) {
      // The unused miniblocks have width bytes but no payload, even when
      // those width bytes are nonzero (Parquet's padding rule).
      if (result.size() == count) break;
      if (width > 64) bad("invalid delta bit width");
      size_t values = block / miniblocks, bytes = values / 8 * width;
      if (bytes > input.size - input.pos) bad("truncated delta miniblock");
      size_t start = input.pos; input.pos += bytes;
      for (size_t i = 0; i < values && result.size() < count; ++i) {
        uint64_t value = 0;
        for (unsigned b = 0; b < width; ++b) {
          size_t bit = i * width + b;
          value |= static_cast<uint64_t>((input.data[start + bit / 8] >> (bit % 8)) & 1) << b;
        }
        value += static_cast<uint64_t>(minimum) + static_cast<uint64_t>(result.back());
        result.push_back(static_cast<int64_t>(value));
      }
    }
  }
  return result;
}
std::vector<std::string> encoded_values(Bytes &input, const Field &field, int encoding, size_t count)
{
  std::vector<std::string> result;
  result.reserve(count);
  if (count == 0) return result;
  if (encoding == 5 && (field.physical == 1 || field.physical == 2)) {
    auto values = delta_values(input, count);
    size_t width = field.physical == 1 ? 4 : 8;
    for (int64_t value : values) {
      std::string bytes(width, '\0');
      for (size_t i = 0; i < width; ++i) bytes[i] = static_cast<char>(static_cast<uint64_t>(value) >> (i * 8));
      Bytes decoded(bytes); result.push_back(plain(decoded, field, 0));
    }
  } else if ((encoding == 6 && field.physical == 6) || (encoding == 7 && (field.physical == 6 || field.physical == 7))) {
    std::vector<int64_t> prefixes;
    if (encoding == 7) prefixes = delta_values(input, count);
    auto lengths = delta_values(input, count);
    std::string previous;
    size_t expanded = count * sizeof(std::string);
    for (size_t i = 0; i < count; ++i) {
      int64_t prefix = prefixes.empty() ? 0 : prefixes[i];
      if (prefix < 0 || static_cast<uint64_t>(prefix) > previous.size() || lengths[i] < 0
          || static_cast<uint64_t>(lengths[i]) > MAX_PAGE - static_cast<uint64_t>(prefix)) bad("invalid delta string length");
      size_t length = static_cast<size_t>(prefix) + static_cast<size_t>(lengths[i]);
      if (length > MAX_MEMORY - expanded) bad("delta string expansion exceeds 256 MiB");
      expanded += length;
      std::string value = previous.substr(0, prefix) + input.string(lengths[i]);
      if (field.physical == 7 && value.size() != static_cast<size_t>(field.length)) bad("delta fixed-width value has wrong length");
      result.push_back(value); previous = std::move(value);
    }
  } else if (encoding == 9 && (field.physical == 1 || field.physical == 2 || field.physical == 4 || field.physical == 5 || field.physical == 7)) {
    size_t width = field.physical == 7 ? field.length : (field.physical == 1 || field.physical == 4) ? 4 : 8;
    if (width > (input.size - input.pos) / count) bad("truncated byte-stream-split values");
    for (size_t i = 0; i < count; ++i) {
      std::string bytes(width, '\0');
      for (size_t b = 0; b < width; ++b) bytes[b] = static_cast<char>(input.data[input.pos + b * count + i]);
      Bytes decoded(bytes); result.push_back(plain(decoded, field, 0));
    }
    input.pos += width * count;
  } else bad("data encoding incompatible with physical type");
  return result;
}

struct Chunk {
  uint64_t start = 0, bytes = 0, values = 0;
  int codec = 0;
};
struct Group { uint64_t rows = 0; std::vector<Chunk> columns; };
struct State {
  uint64_t offset = 0, emitted = 0;
  std::vector<std::string> dictionary;
  std::vector<Cell> page;
  size_t index = 0;
};
}

struct ParquetReader::Impl {
  FILE *input;
  uint64_t file_size = 0, footer_start = 0, row_count = 0, group_row = 0;
  size_t group_index = 0;
  std::vector<Field> fields;
  std::vector<Column> schema;
  std::vector<Group> groups;
  std::vector<size_t> projection;
  std::vector<State> states;
  explicit Impl(FILE *file) : input(file) {
#ifdef _WIN32
    if (_fseeki64(input, 0, SEEK_END) != 0) bad("seek failed");
    auto size = _ftelli64(input);
#else
    if (fseeko(input, 0, SEEK_END) != 0) bad("seek failed");
    auto size = ftello(input);
#endif
    if (size < 12) bad("file is too short");
    file_size = size;
    if (read(0, 4) != "PAR1") bad("invalid file magic");
    std::string trailer = read(file_size - 8, 8);
    Bytes tail(trailer);
    uint64_t footer_size = tail.little(4);
    if (tail.string(4) != "PAR1" || footer_size > 8 * 1024 * 1024 || footer_size > file_size - 12) bad("invalid footer size or magic");
    footer_start = file_size - 8 - footer_size;
    std::string bytes = read(footer_start, footer_size);
    Bytes cursor(bytes); size_t nodes = 0;
    Node metadata = compact(cursor, 12, nodes);
    if (cursor.pos != cursor.size) bad("trailing footer bytes");
    int64_t rows = metadata.at(3).number;
    if (rows < 0) bad("negative row count");
    row_count = rows;
    const auto &elements = metadata.at(2).items;
    if (elements.size() < 2 || elements.size() > 1025 || elements[0].num(5, -1) != static_cast<int64_t>(elements.size() - 1)) bad("only flat schemas are supported");
    for (size_t i = 1; i < elements.size(); ++i) {
      const Node &element = elements[i];
      Field field;
      field.physical = static_cast<int>(element.at(1).number);
      field.converted = static_cast<int>(element.num(6, -1));
      field.length = static_cast<int>(element.num(2, 0));
      if (element.num(5, 0) != 0 || element.num(3, -1) < 0 || element.num(3, -1) > 1) bad("nested or repeated fields are not supported");
      field.column = {element.at(4).text, element.at(4).text, Type::Varchar, element.at(3).number == 1};
      if (field.column.name.empty() || field.column.name.size() > 256) bad("invalid column name");
      if (field.physical == 0) field.column.type = Type::Boolean;
      else if (field.physical == 1 || field.physical == 2) field.column.type = field.converted == 14 ? Type::Varchar : Type::Bigint;
      else if (field.physical == 3) field.column.type = Type::Datetime;
      else if (field.physical == 4 || field.physical == 5) field.column.type = Type::Double;
      else if (field.physical != 6 && field.physical != 7) bad("unsupported physical type");
      if (field.physical == 7 && (field.length <= 0 || field.length > static_cast<int>(MAX_PAGE))) bad("invalid fixed byte-array length");
      if (field.converted == 6) field.column.type = Type::Date;
      if (field.converted == 9 || field.converted == 10) { field.column.type = Type::Datetime; field.timestamp_unit = field.converted == 9 ? 1 : 2; }
      if (field.converted == 5) bad("decimal logical types are not supported");
      if (element.has(10)) {
        const Node &logical = element.at(10);
        if (logical.has(6)) field.column.type = Type::Date;
        if (logical.has(8)) {
          field.column.type = Type::Datetime;
          const Node &unit = logical.at(8).at(2);
          field.timestamp_unit = unit.has(1) ? 1 : unit.has(2) ? 2 : unit.has(3) ? 3 : 0;
          if (field.timestamp_unit == 0) bad("invalid timestamp unit");
        }
        if (logical.has(11)) field.column.type = Type::Null;
      }
      if (field.column.type == Type::Datetime) field.column.scale = 6;
      schema.push_back(field.column); fields.push_back(std::move(field));
    }
    uint64_t summed_rows = 0;
    for (const Node &group_node : metadata.at(4).items) {
      Group group;
      if (group_node.at(3).number < 0) bad("negative row-group size");
      group.rows = group_node.at(3).number;
      if (group.rows > row_count - summed_rows) bad("row-group count exceeds footer count");
      summed_rows += group.rows;
      const auto &chunks = group_node.at(1).items;
      if (chunks.size() != fields.size()) bad("row-group column count differs from schema");
      for (size_t i = 0; i < chunks.size(); ++i) {
        if (chunks[i].has(1) && !chunks[i].at(1).text.empty()) bad("external column chunks are not supported");
        const Node &meta = chunks[i].at(3);
        if (meta.at(1).number != fields[i].physical || meta.at(3).items.size() != 1
            || meta.at(3).items[0].text != fields[i].column.key || meta.at(5).number < 0
            || static_cast<uint64_t>(meta.at(5).number) != group.rows) bad("column metadata differs from schema");
        int64_t offset = meta.at(9).number;
        if (meta.has(11)) offset = std::min(offset, meta.at(11).number);
        int64_t length = meta.at(7).number;
        if (offset < 4 || length < 0 || static_cast<uint64_t>(offset) > footer_start
            || static_cast<uint64_t>(length) > footer_start - offset) bad("column chunk outside file data");
        group.columns.push_back({static_cast<uint64_t>(offset), static_cast<uint64_t>(length), group.rows, static_cast<int>(meta.at(4).number)});
      }
      groups.push_back(std::move(group));
    }
    if (summed_rows != row_count) bad("row-group and footer row counts differ");
    for (size_t i = 0; i < fields.size(); ++i) projection.push_back(i);
    reset();
  }
  std::string read(uint64_t offset, size_t size) {
    if (offset > file_size || size > file_size - offset || size > MAX_PAGE) bad("read outside file bounds");
#ifdef _WIN32
    if (_fseeki64(input, static_cast<int64_t>(offset), SEEK_SET) != 0) bad("seek failed");
#else
    if (fseeko(input, static_cast<off_t>(offset), SEEK_SET) != 0) bad("seek failed");
#endif
    std::string result(size, '\0');
    if (fread(result.data(), 1, size, input) != size) bad("short read");
    return result;
  }
  void reset() { group_index = 0; group_row = 0; states.clear(); }
  size_t memory_limit() {
    size_t bytes = 0;
    for (const auto &state : states) {
      bytes += state.dictionary.capacity() * sizeof(std::string) + state.page.capacity() * sizeof(Cell);
      for (const auto &value : state.dictionary) bytes += value.capacity();
      for (const auto &value : state.page) if (value) bytes += value->capacity();
      if (bytes > MAX_MEMORY) bad("projected page buffers exceed 256 MiB");
    }
    return bytes;
  }
  void load(size_t column) {
    State &state = states[column];
    const Chunk &chunk = groups[group_index].columns[column];
    const Field &field = fields[column];
    state.page.clear(); state.index = 0;
    while (state.page.empty()) {
      if (state.offset >= chunk.start + chunk.bytes) bad("missing data page");
      size_t available = static_cast<size_t>(std::min<uint64_t>(1024 * 1024, chunk.start + chunk.bytes - state.offset));
      std::string header_bytes = read(state.offset, available);
      Bytes header_cursor(header_bytes); size_t nodes = 0;
      Node header = compact(header_cursor, 12, nodes);
      int64_t compressed = header.at(3).number, uncompressed = header.at(2).number;
      state.offset += header_cursor.pos;
      if (compressed < 0 || uncompressed < 0 || compressed > static_cast<int64_t>(MAX_PAGE)
          || uncompressed > static_cast<int64_t>(MAX_PAGE) || static_cast<uint64_t>(compressed) > chunk.start + chunk.bytes - state.offset)
        bad("invalid page size");
      std::string body = read(state.offset, compressed); state.offset += compressed;
      if (header.has(4) && crc32(body) != static_cast<uint32_t>(header.at(4).number)) bad("page checksum mismatch");
      int page_type = static_cast<int>(header.at(1).number);
      if (page_type == 2) {
        if (!state.dictionary.empty() || header.at(7).at(2).number != 0) bad("invalid dictionary page");
        int64_t count = header.at(7).at(1).number;
        if (count < 0 || count > static_cast<int64_t>(MAX_PAGE_VALUES)) bad("dictionary resource limit exceeded");
        std::string decoded = unpack(body, uncompressed, chunk.codec); Bytes values(decoded);
        for (size_t i = 0; i < static_cast<size_t>(count); ++i) state.dictionary.push_back(plain(values, field, i));
        memory_limit(); continue;
      }
      if (page_type != 0 && page_type != 3) continue;
      const Node &data_header = header.at(page_type == 0 ? 5 : 8);
      int64_t count = data_header.at(1).number;
      int encoding = static_cast<int>(data_header.at(page_type == 0 ? 2 : 4).number);
      if (count <= 0 || count > static_cast<int64_t>(MAX_PAGE_VALUES)
          || static_cast<uint64_t>(count) > chunk.values - state.emitted) bad("invalid data page value count");
      std::string decoded;
      size_t definition_bytes = 0;
      if (page_type == 0) decoded = unpack(body, uncompressed, chunk.codec);
      else {
        int64_t definitions = data_header.at(5).number, repetitions = data_header.at(6).number;
        if (definitions < 0 || repetitions != 0 || static_cast<size_t>(definitions) > body.size()
            || definitions > uncompressed) bad("invalid flat v2 level lengths");
        definition_bytes = definitions;
        decoded = body.substr(0, definition_bytes);
        std::string values = body.substr(definition_bytes);
        decoded += data_header.num(7, 1) ? unpack(values, uncompressed - definitions, chunk.codec)
                                         : unpack(values, uncompressed - definitions, 0);
      }
      Bytes cursor(decoded);
      std::vector<uint32_t> definitions(static_cast<size_t>(count), 1);
      if (field.column.nullable) {
        if (page_type == 0) {
          if (data_header.at(3).number == 3) definition_bytes = cursor.little(4);
          else if (data_header.at(3).number == 4) definition_bytes = (static_cast<size_t>(count) + 7) / 8;
          else bad("unsupported definition level encoding");
        }
        if (definition_bytes > cursor.size - cursor.pos) bad("truncated definition levels");
        Bytes levels(cursor.data + cursor.pos, definition_bytes);
        if (page_type == 0 && data_header.at(3).number == 4) {
          for (size_t i = 0; i < static_cast<size_t>(count); ++i) definitions[i] = (levels.data[i / 8] >> (i % 8)) & 1;
        } else definitions = hybrid(levels, 1, count);
        cursor.pos += definition_bytes;
      } else if (page_type == 3 && definition_bytes != 0) bad("required field has definition levels");
      size_t nonnull = std::count(definitions.begin(), definitions.end(), 1u);
      std::vector<uint32_t> indices;
      std::vector<std::string> values;
      if (encoding == 2 || encoding == 8) {
        if (nonnull != 0) { unsigned width = cursor.byte(); indices = hybrid(cursor, width, nonnull); }
      } else if (encoding == 3 && field.physical == 0) {
        if (nonnull != 0) {
          size_t length = cursor.little(4);
          if (length > cursor.size - cursor.pos) bad("truncated boolean RLE stream");
          Bytes values(cursor.data + cursor.pos, length);
          indices = hybrid(values, 1, nonnull); cursor.pos += length;
        }
      } else if (encoding == 5 || encoding == 6 || encoding == 7 || encoding == 9) {
        values = encoded_values(cursor, field, encoding, nonnull);
      } else if (encoding != 0) bad("unsupported data encoding: " + std::to_string(encoding));
      size_t value_index = 0, buffered = memory_limit();
      for (size_t i = 0; i < static_cast<size_t>(count); ++i) {
        Cell value;
        if (definitions[i] == 0) {
        } else if (definitions[i] != 1) bad("invalid flat definition level");
        else if (encoding == 2 || encoding == 8) {
          if (indices[value_index] >= state.dictionary.size()) bad("dictionary index outside dictionary");
          const auto &entry = state.dictionary[indices[value_index++]];
          if (entry.size() + sizeof(Cell) > MAX_MEMORY - buffered) bad("dictionary expansion exceeds 256 MiB");
          value = entry;
        } else if (encoding == 3) value = indices[value_index++] ? "true" : "false";
        else if (encoding == 5 || encoding == 6 || encoding == 7 || encoding == 9) value = std::move(values[value_index++]);
        else value = plain(cursor, field, value_index++);
        size_t bytes = sizeof(Cell) + (value ? value->capacity() : 0);
        if (bytes > MAX_MEMORY - buffered) bad("page expansion exceeds 256 MiB");
        buffered += bytes;
        state.page.emplace_back(std::move(value));
      }
      memory_limit();
    }
  }
  bool next(Row &row) {
    while (group_index < groups.size() && group_row == groups[group_index].rows) { ++group_index; group_row = 0; states.clear(); }
    if (group_index == groups.size()) return false;
    if (states.empty()) {
      states.resize(fields.size());
      for (size_t i = 0; i < fields.size(); ++i) states[i].offset = groups[group_index].columns[i].start;
    }
    row.assign(fields.size(), std::nullopt);
    for (size_t column : projection) {
      State &state = states[column];
      if (state.index == state.page.size()) load(column);
      row[column] = state.page[state.index++]; ++state.emitted;
    }
    ++group_row; return true;
  }
};
ParquetReader::ParquetReader(FILE *file) : impl_(new Impl(file)) {}
ParquetReader::~ParquetReader() = default;
const std::vector<Column> &ParquetReader::columns() const { return impl_->schema; }
uint64_t ParquetReader::rows() const { return impl_->row_count; }
void ParquetReader::project(const std::vector<size_t> &columns) {
  if (impl_->group_row != 0 || impl_->group_index != 0) bad("projection changed after scan started");
  std::vector<size_t> selected = columns;
  std::sort(selected.begin(), selected.end()); selected.erase(std::unique(selected.begin(), selected.end()), selected.end());
  if (!selected.empty() && selected.back() >= impl_->fields.size()) bad("projected column outside schema");
  impl_->projection = std::move(selected);
}
bool ParquetReader::next(Row &row) { return impl_->next(row); }
void ParquetReader::rescan() { impl_->reset(); }
}}}
