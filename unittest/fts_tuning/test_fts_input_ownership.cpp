/* Copyright (c) 2025 OceanBase.
 * Licensed under the Apache License, Version 2.0.
 */

#include <gtest/gtest.h>
#include <cstring>
#include <map>
#include <string>
#include "data_plane/blocksstable/ob_datum_row.h"
#include "data_plane/transaction/ob_tx_desc_lifecycle.h"
#include "sql/das/ob_das_domain_utils.h"

namespace oceanbase
{
namespace sql
{

class FTSInputOwnership : public testing::Test
{
protected:
  static void SetUpTestCase()
  {
    ASSERT_EQ(common::OB_SUCCESS, storage::ObFTParseData::init_global());
  }

  static void TearDownTestCase()
  {
    storage::ObFTParseData::deinit_global();
  }

  void check_rows(ObFTIndexRowCache &cache, const int word_index,
                  const std::map<std::string, uint64_t> &expected)
  {
    std::map<std::string, uint64_t> actual;
    blocksstable::ObDatumRow *row = nullptr;
    int ret = common::OB_SUCCESS;
    while (common::OB_SUCCESS == (ret = cache.get_next_row(row))) {
      ASSERT_NE(nullptr, row);
      const common::ObString word = row->storage_datums_[word_index].get_string();
      EXPECT_TRUE(actual.emplace(std::string(word.ptr(), word.length()),
                                 row->storage_datums_[2].get_uint()).second);
    }
    EXPECT_EQ(common::OB_ITER_END, ret);
    EXPECT_EQ(expected, actual);
  }
};

TEST_F(FTSInputOwnership, CachedIndexRowsOutliveMutableInputAndReuse)
{
  // A mutable caller-owned buffer makes the old dangling/borrowed-input bug
  // deterministic without relying on whether a freed LOB page gets recycled.
  for (const char *parser : {"space.1", "beng.1"}) {
    for (const bool inverted : {false, true}) {
      SCOPED_TRACE(parser);
      SCOPED_TRACE(inverted);
      ObFTIndexRowCache cache;
      ASSERT_EQ(common::OB_SUCCESS, cache.init(inverted, common::ObString::make_string(parser), common::ObString()));
      common::ObObjMeta meta;
      meta.set_varchar();
      meta.set_collation_type(common::CS_TYPE_UTF8MB4_GENERAL_CI);
      const uint64_t doc_id_value = 77;
      common::ObDatum doc_id(reinterpret_cast<const char *>(&doc_id_value), sizeof(doc_id_value), false);
      char input[] = "quartz zephyr quartz";
      ASSERT_EQ(common::OB_SUCCESS, cache.segment(meta, doc_id, common::ObString::make_string(input)));
      std::memset(input, 'x', sizeof(input) - 1);
      check_rows(cache, inverted ? 0 : 1, {{"quartz", 2}, {"zephyr", 1}});

      char next[] = "cobalt orchid";
      ASSERT_EQ(common::OB_SUCCESS, cache.segment(meta, doc_id, common::ObString::make_string(next)));
      std::memset(next, 'y', sizeof(next) - 1);
      check_rows(cache, inverted ? 0 : 1, {{"cobalt", 1}, {"orchid", 1}});
      EXPECT_EQ(common::OB_ITER_END, cache.segment(meta, doc_id, common::ObString()));
      check_rows(cache, inverted ? 0 : 1, {});
    }
  }
}

TEST_F(FTSInputOwnership, BENGCopiesReusedNormalizerBuffers)
{
  ObFTIndexRowCache cache;
  ASSERT_EQ(common::OB_SUCCESS,
      cache.init(true, common::ObString::make_string("beng.1"), common::ObString()));
  common::ObObjMeta meta;
  meta.set_varchar();
  meta.set_collation_type(common::CS_TYPE_UTF8MB4_GENERAL_CI);
  const uint64_t doc_id_value = 88;
  common::ObDatum doc_id(reinterpret_cast<const char *>(&doc_id_value), sizeof(doc_id_value), false);
  char input[] = "QUARTZ zephyr ORCHID cobalt QUARTZ";
  ASSERT_EQ(common::OB_SUCCESS, cache.segment(meta, doc_id, common::ObString::make_string(input)));
  std::memset(input, 'x', sizeof(input) - 1);
  check_rows(cache, 0, {{"quartz", 2}, {"zephyr", 1}, {"orchid", 1}, {"cobalt", 1}});
}

TEST_F(FTSInputOwnership, RepeatedDocumentsMatchIndependentRows)
{
  using WordCounts = std::map<std::string, std::pair<uint64_t, uint64_t>>;
  // Alternate bucket sizes, repeated words, collations, empty/stopword-only
  // documents, and failed segmentation. No word/count from an earlier document
  // may survive, including after reset/re-init.
  for (const char *parser : {"space.1", "beng.1", "ik.1"}) {
    for (const bool inverted : {false, true}) {
      SCOPED_TRACE(parser);
      SCOPED_TRACE(inverted);
      ObFTIndexRowCache cache;
      ASSERT_EQ(common::OB_SUCCESS, cache.init(inverted, common::ObString::make_string(parser), common::ObString()));
      for (int round = 0; round < 72; ++round) {
        SCOPED_TRACE(round);
        std::string input;
        switch (round % 6) {
          case 0: input = "QUARTZ zephyr QUARTZ"; break;
          case 1:
            // Exceed a normal node block, then shrink to an empty/small
            // document. Alphabetic suffixes remain distinct for BENG too.
            for (int i = 0; i < 300; ++i) {
              input += "quartz";
              input += char('a' + i / (26 * 26));
              input += char('a' + (i / 26) % 26);
              input += char('a' + i % 26);
              input += ' ';
            }
            break;
          case 2: input = ""; break;
          case 3: input = "the and or of"; break;
          case 4: input = "zephyr orchid cobalt zephyr"; break;
          case 5: input = "quartz"; break;
        }
        common::ObObjMeta meta;
        meta.set_varchar();
        meta.set_collation_type(round % 3 == 0 ? common::CS_TYPE_UTF8MB4_BIN : common::CS_TYPE_UTF8MB4_GENERAL_CI);
        const uint64_t doc_id_value = 900 + round;
        common::ObDatum doc_id(reinterpret_cast<const char *>(&doc_id_value), sizeof(doc_id_value), false);
        common::ObArenaAllocator reference_allocator("FTWordMapTest");
        storage::ObFTParseHelper reference_helper;
        ASSERT_EQ(common::OB_SUCCESS,
            reference_helper.init(&reference_allocator, common::ObString::make_string(parser), common::ObString()));
        ObDomainIndexRow reference_rows;
        const common::ObString text(input.size(), input.data());
        const int expected_ret = ObDASDomainUtils::generate_fulltext_word_rows(
            reference_allocator, &reference_helper, meta, doc_id, text, inverted, reference_rows);
        if (0 == std::strcmp(parser, "ik.1")) {
          // This standalone fixture initializes parser metadata, not the server
          // KV dictionary cache. The fresh IK path returns exactly OB_NOT_INIT
          // for nonempty input. Check this failure precisely for both paths;
          // successful dictionary-backed IK is covered by native SQL tests.
          ASSERT_EQ(input.empty() ? common::OB_ITER_END : common::OB_NOT_INIT, expected_ret);
        } else {
          ASSERT_TRUE(common::OB_SUCCESS == expected_ret || common::OB_ITER_END == expected_ret)
              << "fresh-map parser result=" << expected_ret;
        }
        ASSERT_EQ(expected_ret, cache.segment(meta, doc_id, text));
        WordCounts expected;
        const int word_index = inverted ? 0 : 1;
        for (int64_t i = 0; i < reference_rows.count(); ++i) {
          const blocksstable::ObStorageDatum *cells = reference_rows.at(i)->storage_datums_;
          const common::ObString word = cells[word_index].get_string();
          ASSERT_TRUE(expected.emplace(std::string(word.ptr(), word.length()),
              std::make_pair(cells[2].get_uint(), cells[3].get_uint())).second);
        }
        WordCounts actual;
        blocksstable::ObDatumRow *row = nullptr;
        int ret = common::OB_SUCCESS;
        while (common::OB_SUCCESS == (ret = cache.get_next_row(row))) {
          ASSERT_NE(nullptr, row);
          const common::ObString word = row->storage_datums_[word_index].get_string();
          const common::ObString row_doc_id = row->storage_datums_[inverted ? 1 : 0].get_string();
          ASSERT_EQ(sizeof(doc_id_value), row_doc_id.length());
          EXPECT_EQ(0, std::memcmp(row_doc_id.ptr(), &doc_id_value, sizeof(doc_id_value)));
          ASSERT_TRUE(actual.emplace(std::string(word.ptr(), word.length()),
              std::make_pair(row->storage_datums_[2].get_uint(), row->storage_datums_[3].get_uint())).second);
        }
        EXPECT_EQ(common::OB_ITER_END, ret);
        EXPECT_EQ(expected, actual);
        if (round % 12 == 11) {
          cache.reset();
          ASSERT_EQ(common::OB_SUCCESS,
              cache.init(inverted, common::ObString::make_string(parser), common::ObString()));
        } else {
          common::ObObjMeta invalid_meta;
          invalid_meta.set_varchar();
          invalid_meta.set_collation_type(common::CS_TYPE_INVALID);
          EXPECT_EQ(common::OB_INVALID_ARGUMENT,
              cache.segment(invalid_meta, doc_id, common::ObString::make_string("stale")));
          EXPECT_EQ(common::OB_ITER_END, cache.get_next_row(row));
        }
      }
    }
  }
}

} // namespace sql
} // namespace oceanbase

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
