/* Copyright (c) 2025 OceanBase.
 * Licensed under the Apache License, Version 2.0.
 */
#include <gtest/gtest.h>
#include <array>
#include <cstring>
#include <vector>
#include "lib/allocator/page_arena.h"
#include "storage/blocksstable/encoding/ob_integer_base_diff_decoder.h"

namespace oceanbase
{
namespace blocksstable
{
namespace
{

void check_integer_batch(const int width, const int rows, const uint64_t base,
    const common::ObObjType type, const bool nullable, const bool packed)
{
  common::ObArenaAllocator arena("IntBatchTest");
  ObMicroBlockHeader micro;
  micro.row_count_ = rows;
  micro.extend_value_bit_ = nullable ? 1 : 0;
  micro.rowkey_column_count_ = 0;
  ObColumnHeader col;
  col.type_ = ObColumnHeader::INTEGER_BASE_DIFF;
  col.obj_type_ = type;
  col.offset_ = 0;
  const int store_bytes = get_type_size_map()[type];
  col.length_ = sizeof(ObIntegerBaseDiffHeader) + store_bytes;
  col.set_fix_lenght_attr();
  if (nullable) { col.set_has_extend_value_attr(); }
  if (packed) { col.set_bit_packing_attr(); }
  const int null_bits = nullable ? rows : 0;
  const int stride_bits = packed ? width : width * 8;
  const int offset_bits = packed ? null_bits : ((null_bits + 7) / 8) * 8;
  std::vector<char> encoded(col.length_ + (offset_bits + rows * stride_bits + 7) / 8, 0);
  auto *header = new (encoded.data()) ObIntegerBaseDiffHeader();
  header->length_ = width;
  std::memcpy(encoded.data() + sizeof(*header), &base, store_bytes);
  unsigned char *data = reinterpret_cast<unsigned char *>(encoded.data() + col.length_);
  const uint64_t mask = stride_bits == 64 ? UINT64_MAX : (uint64_t(1) << stride_bits) - 1;
  for (int r = 0; r < rows; ++r) {
    if (nullable && r % 3 == 1) { data[r / 8] |= 1 << (r % 8); }
    const uint64_t delta = (uint64_t(r) * UINT64_C(0x9e3779b97f4a7c15)) & mask;
    for (int b = 0; b < stride_bits; ++b) {
      if ((delta >> b) & 1) {
        const int p = offset_bits + r * stride_bits + b;
        data[p / 8] |= 1 << (p % 8);
      }
    }
  }
  const std::vector<char> original = encoded;
  common::ObObjMeta meta;
  meta.set_type(type);
  ObColumnDecoderCtx ctx;
  ctx.fill(meta, &micro, &col, &arena, 0);
  ObIntegerBaseDiffDecoder decoder;
  ASSERT_EQ(common::OB_SUCCESS, decoder.init(micro, col, encoded.data()));
  uint32_t datum_len = 0;
  ASSERT_EQ(common::OB_SUCCESS,
      get_uint_data_datum_len(common::ObDatum::get_obj_datum_map_type(type), datum_len));
  std::vector<int32_t> ids;
  // Last row first exercises the tail-reader bounds. Include duplicates and a
  // reverse-order batch; a contiguous-only decoder must fail this test.
  for (int r = rows - 1; r >= 0; --r) { ids.push_back(r); }
  ids.push_back(rows - 1);
  ids.push_back(0);
  std::vector<common::ObDatum> batch(ids.size());
  std::vector<std::array<char, 16>> destinations(ids.size());
  for (size_t i = 0; i < ids.size(); ++i) {
    destinations[i].fill(char(0x5a));
    batch[i].ptr_ = destinations[i].data();
    batch[i].pack_ = 0;
  }
  ASSERT_EQ(common::OB_SUCCESS,
      decoder.batch_decode(ctx, nullptr, ids.data(), nullptr, ids.size(), batch.data()));
  ObBitStream bit_stream;
  for (size_t i = 0; i < ids.size(); ++i) {
    std::array<char, 16> scalar_bytes;
    scalar_bytes.fill(char(0x5a));
    common::ObDatum scalar;
    scalar.ptr_ = scalar_bytes.data();
    scalar.pack_ = 0;
    ASSERT_EQ(common::OB_SUCCESS, decoder.decode(ctx, scalar, ids[i], bit_stream,
        encoded.data(), encoded.size()));
    const bool is_null = nullable && ids[i] % 3 == 1;
    ASSERT_EQ(is_null, batch[i].is_null());
    ASSERT_EQ(scalar.pack_, batch[i].pack_);
    EXPECT_EQ(scalar_bytes, destinations[i]);
    if (!is_null) {
      const uint64_t delta = (uint64_t(ids[i]) * UINT64_C(0x9e3779b97f4a7c15)) & mask;
      const uint64_t expected = delta + base;
      EXPECT_EQ(0, std::memcmp(destinations[i].data(), &expected, datum_len));
    }
    for (size_t b = datum_len; b < destinations[i].size(); ++b) {
      EXPECT_EQ(char(0x5a), destinations[i][b]);
    }
  }
  EXPECT_EQ(original, encoded);
  // Empty batches must not inspect row IDs or write output.
  EXPECT_EQ(common::OB_SUCCESS, decoder.batch_decode(ctx, nullptr, nullptr, nullptr, 0, nullptr));
}

TEST(IntegerBatchDecoder, AllBitWidthsTailRowsAndModularBases)
{
  for (int width = 1; width <= 64; ++width) {
    for (const int rows : {1, 2, 7, 8, 9, 16, 17, 33}) {
      for (const uint64_t base : {UINT64_C(0), UINT64_C(7), UINT64_MAX,
                                 UINT64_C(0x8000000000000000)}) {
        SCOPED_TRACE(testing::Message() << width << "/" << rows << "/" << base);
        check_integer_batch(width, rows, base, common::ObUInt64Type, false, true);
      }
    }
  }
}

TEST(IntegerBatchDecoder, SignedNullAndNonpackedFallbacks)
{
  for (const int width : {1, 7, 8, 9, 10, 16, 25, 26, 32, 63, 64}) {
    check_integer_batch(width, 17, UINT64_MAX - 31, common::ObIntType, false, true);
    check_integer_batch(width, 17, UINT64_C(9), common::ObUInt64Type, true, true);
  }
  for (const int bytes : {1, 2, 4, 8}) {
    check_integer_batch(bytes, 17, UINT64_C(3), common::ObUInt64Type, false, false);
    check_integer_batch(bytes, 17, UINT64_C(3), common::ObUInt64Type, true, false);
  }
  for (const auto type : {common::ObTinyIntType, common::ObSmallIntType, common::ObInt32Type}) {
    check_integer_batch(7, 17, UINT64_MAX - 31, type, false, true);
  }
  for (const int width : {1, 7, 9, 16, 25, 32}) {
    check_integer_batch(width, 9, UINT64_C(0), common::ObFloatType, false, true);
  }
}

TEST(IntegerBatchDecoder, InitializationErrorIsPreserved)
{
  ObIntegerBaseDiffDecoder decoder;
  ObColumnDecoderCtx ctx;
  EXPECT_EQ(common::OB_NOT_INIT,
      decoder.batch_decode(ctx, nullptr, nullptr, nullptr, 0, nullptr));
}

} // namespace
} // namespace blocksstable
} // namespace oceanbase

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
