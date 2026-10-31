/* Copyright (c) 2025 OceanBase.
 * Licensed under the Apache License, Version 2.0.
 */

#include <gtest/gtest.h>
#include <array>
#include <cstring>
#include <string>

// Include dependencies before exposing the one private comparison entry point.
// The test then calls the production inline function, not a copied algorithm.
#include "lib/charset/ob_charset.h"
#include "query/ob_sql_define.h"
#include "storage/retrieval/ob_inv_idx_param_estimator.h"
#include "storage/retrieval/ob_sparse_bmw_iter.h"
#include "storage/retrieval/ob_sparse_daat_iter.h"
#include "storage/retrieval/ob_text_retrieval_token_iter.h"
#define private public
#include "storage/retrieval/ob_text_daat_iter.h"
#undef private

namespace oceanbase
{
namespace storage
{

class DocOnlyBinaryCompare : public ::testing::Test
{
protected:
  void SetUp() override
  {
    iter_.doc_only_charset_ = common::ObCharset::get_charset(common::CS_TYPE_BINARY);
    ASSERT_NE(nullptr, iter_.doc_only_charset_);
    comparator_ = common::ObDatumFuncs::get_basic_func(
        common::ObVarcharType, common::CS_TYPE_BINARY)->null_first_cmp_;
    ASSERT_NE(nullptr, comparator_);
    iter_.doc_only_cmp_ = comparator_;
  }

  void check(const common::ObDatum &left, const common::ObDatum &right)
  {
    int expected = 99;
    int actual = 88;
    ASSERT_EQ(common::OB_SUCCESS, comparator_(left, right, expected, nullptr));
    ASSERT_EQ(common::OB_SUCCESS, iter_.compare_doc_only_ids(left, right, actual));
    EXPECT_EQ(expected, actual);
  }

  void check_strings(const char *left, uint32_t left_size,
                     const char *right, uint32_t right_size)
  {
    check(common::ObDatum(left, left_size, false),
          common::ObDatum(right, right_size, false));
  }

  ObTextDaaTIter iter_;
  common::ObDatumCmpFuncType comparator_ = nullptr;
};

TEST_F(DocOnlyBinaryCompare, EveryByteAndUnsignedOrdering)
{
  // Offsets deliberately make the two 64-bit chunks unaligned. Bytes after
  // the first difference have reverse order, so a wrong word order fails.
  std::array<char, 19> left{};
  std::array<char, 21> right{};
  const unsigned char edges[] = {0, 1, 0x20, 0x7f, 0x80, 0xfe, 0xff};
  for (int position = 0; position < 16; ++position) {
    for (unsigned char l : edges) {
      for (unsigned char r : edges) {
        std::memset(left.data() + 1, 0x80, 16);
        std::memset(right.data() + 3, 0x80, 16);
        left[position + 1] = static_cast<char>(l);
        right[position + 3] = static_cast<char>(r);
        if (l != r && position < 15) {
          left[position + 2] = static_cast<char>(l < r ? 0xff : 0);
          right[position + 4] = static_cast<char>(l < r ? 0 : 0xff);
        }
        check_strings(left.data() + 1, 16, right.data() + 3, 16);
      }
    }
  }
}

TEST_F(DocOnlyBinaryCompare, RandomEqualAndSharedPrefixKeys)
{
  // Reproducible pseudorandom bytes, with forced prefixes exercising both
  // words and exact equality. No memory outside these 16 bytes may be read.
  std::array<char, 16> left{};
  std::array<char, 16> right{};
  uint64_t state = UINT64_C(0x687261646f6c6274);
  for (int trial = 0; trial < 100000; ++trial) {
    for (int byte = 0; byte < 16; ++byte) {
      state ^= state << 13;
      state ^= state >> 7;
      state ^= state << 17;
      left[byte] = static_cast<char>(state);
      right[byte] = static_cast<char>(state >> 8);
    }
    const int common_prefix = trial % 17;
    std::memcpy(right.data(), left.data(), common_prefix);
    check_strings(left.data(), 16, right.data(), 16);
  }
}

TEST_F(DocOnlyBinaryCompare, LengthTrailingBytesNullAndTypedFallback)
{
  std::string left(40, '\0');
  std::string right(40, '\0');
  for (int byte = 0; byte < 40; ++byte) {
    left[byte] = right[byte] = static_cast<char>(byte % 3 == 0 ? 0x20 : 0x80);
  }
  for (uint32_t left_size = 0; left_size <= 40; ++left_size) {
    for (uint32_t right_size = 0; right_size <= 40; ++right_size) {
      check_strings(left.data(), left_size, right.data(), right_size);
    }
  }
  common::ObDatum null_datum(left.data(), 16, false);
  null_datum.set_null();
  common::ObDatum other(right.data(), 16, false);
  check(null_datum, other);
  check(other, null_datum);
  check(null_datum, null_datum);
  // An unavailable cached charset must retain the typed comparator branch.
  iter_.doc_only_charset_ = nullptr;
  check(other, other);
  right[15] = static_cast<char>(0xff);
  check_strings(left.data(), 16, right.data(), 16);
}

} // namespace storage
} // namespace oceanbase

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
