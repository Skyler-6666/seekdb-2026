/* Copyright (c) 2025 OceanBase.
 * Licensed under the Apache License, Version 2.0.
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <cstring>
#include <string>
#include <vector>
#include "lib/allocator/ob_allocator.h"
#include "lib/allocator/page_arena.h"
#include "storage/blocksstable/encoding/ob_string_diff_decoder.h"

namespace oceanbase
{
namespace blocksstable
{
namespace
{

class RecordingAllocator : public common::ObWrapperAllocator
{
public:
  RecordingAllocator() : common::ObWrapperAllocator(&arena_), arena_("StrDiffTest") {}
  using common::ObWrapperAllocator::alloc;
  void *alloc(const int64_t size) override
  {
    last_request_ = size;
    return fail_ ? nullptr : arena_.alloc(size);
  }
  common::ObArenaAllocator arena_;
  int64_t last_request_ = 0;
  bool fail_ = false;
};

struct EncodedFixedColumn
{
  static constexpr int64_t ROWS = 8;
  std::vector<char> bytes_;
  std::vector<std::string> expected_;
  ObMicroBlockHeader micro_;
  ObColumnHeader col_;
  ObColumnDecoderCtx ctx_;

  EncodedFixedColumn(const int width, const uint32_t diff_mask,
                     RecordingAllocator &allocator,
                     const bool has_null = false, const bool hex = false)
  {
    std::vector<ObStringDiffHeader::DiffDesc> descriptors;
    std::vector<unsigned char> common_bytes;
    int diff_count = 0;
    for (int p = 0; p < width; ++p) {
      const bool different = 0 != (diff_mask & (uint32_t(1) << p));
      if (descriptors.empty() || bool(descriptors.back().diff_) != different) {
        ObStringDiffHeader::DiffDesc descriptor;
        descriptor.diff_ = different;
        descriptor.count_ = 1;
        descriptors.push_back(descriptor);
      } else {
        descriptors.back().count_++;
      }
      if (different) {
        ++diff_count;
      } else {
        common_bytes.push_back(static_cast<unsigned char>(p * 53));
      }
    }
    const int64_t packed_length = hex ? (diff_count + 1) / 2 : diff_count;
    const int64_t metadata_length = sizeof(ObStringDiffHeader) + descriptors.size()
        + (hex ? 4 : 0) + common_bytes.size();
    const int64_t null_bytes = has_null ? (ROWS + 7) / 8 : 0;
    bytes_.resize(metadata_length + null_bytes + ROWS * packed_length, 0);
    ObStringDiffHeader *header = new (bytes_.data()) ObStringDiffHeader();
    header->string_size_ = width;
    header->diff_desc_cnt_ = descriptors.size();
    header->hex_char_array_size_ = hex ? 4 : 0;
    header->length_ = packed_length;
    for (size_t i = 0; i < descriptors.size(); ++i) {
      header->diff_descs_[i] = descriptors[i];
    }
    const unsigned char hex_values[] = {0, 0x7f, 0x80, 0xff};
    if (hex) {
      std::memcpy(header->hex_char_array(), hex_values, sizeof(hex_values));
    }
    if (!common_bytes.empty()) {
      std::memcpy(header->common_data(), common_bytes.data(), common_bytes.size());
    }
    unsigned char *data = reinterpret_cast<unsigned char *>(bytes_.data() + metadata_length);
    if (has_null) {
      // Rows 1 and 6 are NULL, including a high bit in the bitmap.
      data[0] = (1 << 1) | (1 << 6);
    }
    for (int r = 0; r < ROWS; ++r) {
      std::string expected(width, '\0');
      unsigned char *payload = data + null_bytes + r * packed_length;
      int diff_pos = 0;
      for (int p = 0; p < width; ++p) {
        const bool different = 0 != (diff_mask & (uint32_t(1) << p));
        unsigned char value = static_cast<unsigned char>(p * 53);
        if (different) {
          if (hex) {
            const unsigned char index = (r + p) % 4;
            value = hex_values[index];
            payload[diff_pos / 2] |= index << ((diff_pos % 2 == 0) ? 4 : 0);
          } else {
            value = static_cast<unsigned char>(r * 37 + p * 17);
            payload[diff_pos] = value;
          }
          ++diff_pos;
        }
        expected[p] = static_cast<char>(value);
      }
      expected_.push_back(expected);
    }
    micro_.row_count_ = ROWS;
    micro_.extend_value_bit_ = has_null ? 1 : 0;
    micro_.rowkey_column_count_ = 0;
    col_.offset_ = 0;
    col_.length_ = metadata_length;
    col_.type_ = ObColumnHeader::STRING_DIFF;
    col_.obj_type_ = common::ObVarcharType;
    col_.set_fix_lenght_attr();
    if (has_null) {
      col_.set_has_extend_value_attr();
    }
    common::ObObjMeta meta;
    meta.set_varchar();
    meta.set_collation_type(common::CS_TYPE_BINARY);
    ctx_.fill(meta, &micro_, &col_, &allocator, 0);
  }

  void check(RecordingAllocator &allocator, const std::vector<int32_t> &ids,
             const bool nulls = false)
  {
    ObStringDiffDecoder decoder;
    ASSERT_EQ(common::OB_SUCCESS, decoder.init(micro_, col_, bytes_.data()));
    const std::vector<char> original = bytes_;
    std::vector<common::ObDatum> values(ids.size());
    ASSERT_EQ(common::OB_SUCCESS, decoder.batch_decode(ctx_, nullptr, ids.data(),
        nullptr, ids.size(), values.data()));
    EXPECT_EQ(int64_t(128 * ids.size()), allocator.last_request_);
    for (size_t i = 0; i < ids.size(); ++i) {
      const bool is_null = nulls && (ids[i] == 1 || ids[i] == 6);
      ASSERT_EQ(is_null, values[i].is_null());
      if (!is_null) {
        ASSERT_EQ(expected_[ids[i]].size(), values[i].len_);
        EXPECT_EQ(expected_[ids[i]], std::string(values[i].ptr_, values[i].len_));
        if (i > 0 && !values[i - 1].is_null()) {
          EXPECT_EQ(int64_t(128), values[i].ptr_ - values[i - 1].ptr_);
        }
      }
    }
    // A subsequent decode allocates a new output batch; prior output must live
    // until the caller's arena is reused, just as before the optimization.
    std::vector<common::ObDatum> second(ids.size());
    ASSERT_EQ(common::OB_SUCCESS, decoder.batch_decode(ctx_, nullptr, ids.data(),
        nullptr, ids.size(), second.data()));
    for (size_t i = 0; i < ids.size(); ++i) {
      if (!values[i].is_null()) {
        EXPECT_EQ(expected_[ids[i]], std::string(values[i].ptr_, values[i].len_));
      }
    }
    EXPECT_EQ(original, bytes_);
  }
};

TEST(FixedStringDiffDecoder, Every16ByteDifferencePatternAndScatteredRows)
{
  RecordingAllocator allocator;
  const std::vector<int32_t> ids = {7, 0, 5, 5, 2, 6, 1, 3};
  for (uint32_t mask = 0; mask < (uint32_t(1) << 16); ++mask) {
    SCOPED_TRACE(mask);
    EncodedFixedColumn encoded(16, mask, allocator);
    encoded.check(allocator, ids);
    allocator.arena_.reuse();
  }
}

TEST(FixedStringDiffDecoder, FallbackWidthsScalarNullAndHex)
{
  RecordingAllocator allocator;
  const std::vector<int32_t> ids = {7, 0, 5, 5, 2, 6, 1, 3};
  for (const int width : {0, 1, 15, 17}) {
    SCOPED_TRACE(width);
    EncodedFixedColumn encoded(width, 0xaaaa, allocator);
    encoded.check(allocator, ids);
    allocator.arena_.reuse();
  }
  EncodedFixedColumn scalar(16, 0xff00, allocator);
  scalar.check(allocator, {4});
  EncodedFixedColumn nullable(16, 0x0ff0, allocator, true);
  nullable.check(allocator, ids, true);
  EncodedFixedColumn hex(16, 0x0ff0, allocator, false, true);
  hex.check(allocator, ids);
}

TEST(FixedStringDiffDecoder, AllocationFailurePreservesErrorAndInput)
{
  RecordingAllocator allocator;
  EncodedFixedColumn encoded(16, 0x03c0, allocator);
  ObStringDiffDecoder decoder;
  ASSERT_EQ(common::OB_SUCCESS, decoder.init(encoded.micro_, encoded.col_, encoded.bytes_.data()));
  const std::vector<char> original = encoded.bytes_;
  const int32_t ids[] = {7, 0};
  common::ObDatum datums[2];
  allocator.fail_ = true;
  EXPECT_EQ(common::OB_ALLOCATE_MEMORY_FAILED,
      decoder.batch_decode(encoded.ctx_, nullptr, ids, nullptr, 2, datums));
  EXPECT_EQ(original, encoded.bytes_);
}

} // namespace
} // namespace blocksstable
} // namespace oceanbase

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
