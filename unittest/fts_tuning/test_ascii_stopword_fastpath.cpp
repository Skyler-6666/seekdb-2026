/* Copyright (c) 2025 OceanBase.
 * Licensed under the Apache License, Version 2.0.
 */

#include <gtest/gtest.h>
#include <string>
#include <vector>
#include "lib/allocator/page_arena.h"
#include "storage/fts/ob_fts_stop_word.h"

namespace oceanbase
{
namespace storage
{
namespace
{

// The original public checker's conversion/hash-set path, retained as an oracle
// independently of whether the linked production checker uses the new index.
class LegacyStopWordReference
{
public:
  ~LegacyStopWordReference() { if (set_.created()) { set_.destroy(); } }
  int init()
  {
    int ret = set_.create(37, "RefStopWord", "RefStopWord");
    meta_.set_varchar();
    meta_.set_collation_type(common::CS_TYPE_UTF8MB4_GENERAL_CI);
    for (size_t i = 0; ret == common::OB_SUCCESS
        && i < sizeof(ob_stop_word_list) / sizeof(ob_stop_word_list[0]); ++i) {
      const common::ObString text = common::ObString::make_string(ob_stop_word_list[i]);
      ret = set_.set_refactored(ObFTWord(text.length(), text.ptr(), meta_));
    }
    return ret;
  }
  int check(const ObFTWord &word, bool &is_stopword)
  {
    if (word.empty()) { return common::OB_INVALID_ARGUMENT; }
    common::ObArenaAllocator allocator("RefStopWord");
    common::ObString converted;
    int ret = common::ObCharset::charset_convert(allocator,
        word.get_word().get_string(), word.get_collation_type(),
        common::CS_TYPE_UTF8MB4_GENERAL_CI, converted);
    if (ret == common::OB_SUCCESS) {
      ret = set_.exist_refactored(ObFTWord(converted.length(), converted.ptr(), meta_));
      if (ret == common::OB_HASH_EXIST) {
        is_stopword = true;
        ret = common::OB_SUCCESS;
      } else if (ret == common::OB_HASH_NOT_EXIST) {
        is_stopword = false;
        ret = common::OB_SUCCESS;
      } else if (ret == common::OB_SUCCESS) {
        ret = common::OB_ERR_UNEXPECTED;
      }
    }
    return ret;
  }
private:
  common::hash::ObHashSet<ObFTWord> set_;
  common::ObObjMeta meta_;
};

ObFTWord make_word(const std::string &text, const common::ObCollationType type)
{
  common::ObObjMeta meta;
  meta.set_varchar();
  meta.set_collation_type(type);
  return ObFTWord(text.size(), text.data(), meta);
}

class AsciiStopWordFastpath : public testing::Test
{
protected:
  void SetUp() override
  {
    ASSERT_EQ(common::OB_SUCCESS, checker_.init());
    ASSERT_EQ(common::OB_SUCCESS, reference_.init());
  }
  void check(const std::string &text, const common::ObCollationType type)
  {
    const ObFTWord word = make_word(text, type);
    for (const bool initial : {false, true}) {
      bool expected = initial;
      bool actual = initial;
      const int expected_ret = reference_.check(word, expected);
      EXPECT_EQ(expected_ret, checker_.check_stopword(word, actual));
      EXPECT_EQ(expected, actual);
    }
  }
  ObStopWordChecker checker_;
  LegacyStopWordReference reference_;
};

TEST_F(AsciiStopWordFastpath, DictionaryAllCasesAndAsciiNonMembers)
{
  for (const auto type : {common::CS_TYPE_UTF8MB4_GENERAL_CI,
                         common::CS_TYPE_UTF8MB4_BIN}) {
    SCOPED_TRACE(type);
    for (const auto &entry : ob_stop_word_list) {
      const std::string word(entry);
      for (size_t bits = 0; bits < (size_t(1) << word.size()); ++bits) {
        std::string mixed = word;
        for (size_t i = 0; i < mixed.size(); ++i) {
          if (bits & (size_t(1) << i)) { mixed[i] -= 'a' - 'A'; }
        }
        check(mixed, type);
        check(mixed + "x", type);
        check("x" + mixed, type);
      }
    }
    // Exhaust all 1-, 2- and 3-letter words, including actual and absent entries.
    for (int length = 1, count = 26; length <= 3; ++length, count *= 26) {
      for (int value = 0; value < count; ++value) {
        std::string word(length, 'a');
        int remaining = value;
        for (int i = 0; i < length; ++i) {
          word[i] += remaining % 26;
          remaining /= 26;
        }
        check(word, type);
      }
    }
    uint32_t state = 0x23a19b57;
    for (int i = 0; i < 5000; ++i) {
      std::string word(4 + i % 9, 'a');
      for (char &ch : word) {
        state = state * 1664525U + 1013904223U;
        ch = char('a' + (state >> 16) % 26);
        if (state & 1) { ch -= 'a' - 'A'; }
      }
      check(word, type);
    }
  }
}

TEST_F(AsciiStopWordFastpath, UnicodeWhitespaceBinaryAndOtherCharsetFallbacks)
{
  const std::vector<std::string> utf8_inputs = {
    "", "a", "THE", "where", "quartz", "a ", " a", "\tthe", "the\n", "a!", "1a", "a_b",
    "th\xc3\xa9", "\xc3\xa1", "\xc4\xb0", "a\xcc\x81", "\xef\xbd\x81",
    "\xe4\xb8\xad\xe6\x96\x87", "\xf0\x9f\x8c\x99", std::string("a\0", 2),
    std::string("th\0e", 4), std::string(9, 'a'), std::string(10, 'a'), std::string(80, 'a')
  };
  for (const auto type : {common::CS_TYPE_UTF8MB4_GENERAL_CI,
                         common::CS_TYPE_UTF8MB4_BIN,
                         common::CS_TYPE_UTF8MB4_UNICODE_CI,
                         common::CS_TYPE_BINARY}) {
    SCOPED_TRACE(type);
    ASSERT_TRUE(common::ObCharset::is_valid_collation(type));
    for (const auto &text : utf8_inputs) {
      SCOPED_TRACE(testing::PrintToString(text));
      check(text, type);
      common::ObArenaAllocator allocator("StopWordConvert");
      common::ObString encoded;
      ASSERT_EQ(common::OB_SUCCESS, common::ObCharset::charset_convert(allocator,
          common::ObString(text.size(), text.data()), common::CS_TYPE_UTF8MB4_GENERAL_CI,
          type, encoded));
      check(std::string(encoded.ptr(), encoded.length()), type);
    }
  }
  // These collation enum values exist, but this seekdb branch explicitly
  // supports only the four collations above. Preserve and verify the public
  // API's rejection instead of treating these as successful transcodings.
  for (const auto type : {common::CS_TYPE_UTF16_BIN, common::CS_TYPE_UTF16LE_BIN,
                         common::CS_TYPE_LATIN1_SWEDISH_CI, common::CS_TYPE_GBK_CHINESE_CI}) {
    SCOPED_TRACE(type);
    ASSERT_FALSE(common::ObCharset::is_valid_collation(type));
    for (const auto &text : utf8_inputs) {
      SCOPED_TRACE(testing::PrintToString(text));
      common::ObArenaAllocator allocator("StopWordConvert");
      common::ObString encoded;
      ASSERT_EQ(common::OB_INVALID_ARGUMENT, common::ObCharset::charset_convert(allocator,
          common::ObString(text.size(), text.data()), common::CS_TYPE_UTF8MB4_GENERAL_CI,
          type, encoded));
      EXPECT_TRUE(encoded.empty());
      check(text, type);
    }
  }
  check(std::string("\0A", 2), common::CS_TYPE_UTF16_BIN);
  check(std::string("A\0", 2), common::CS_TYPE_UTF16LE_BIN);
  check(std::string("th\xe9", 3), common::CS_TYPE_LATIN1_SWEDISH_CI);
  check(std::string("\xd6\xd0\xce\xc4", 4), common::CS_TYPE_GBK_CHINESE_CI);
  check(std::string("\xff", 1), common::CS_TYPE_UTF8MB4_GENERAL_CI);
  check(std::string("\xc3", 1), common::CS_TYPE_UTF8MB4_BIN);
  check("a", common::CS_TYPE_INVALID);
}

TEST(AsciiStopWordLifecycle, ErrorsKeepOutputAndReinitializationUsesSameDictionary)
{
  ObStopWordChecker checker;
  const std::string text = "THE";
  const ObFTWord word = make_word(text, common::CS_TYPE_UTF8MB4_GENERAL_CI);
  for (const bool initial : {false, true}) {
    bool output = initial;
    EXPECT_EQ(common::OB_NOT_INIT, checker.check_stopword(word, output));
    EXPECT_EQ(initial, output);
  }
  for (int cycle = 0; cycle < 3; ++cycle) {
    ASSERT_EQ(common::OB_SUCCESS, checker.init());
    EXPECT_EQ(common::OB_INIT_TWICE, checker.init());
    bool output = false;
    ASSERT_EQ(common::OB_SUCCESS, checker.check_stopword(word, output));
    EXPECT_TRUE(output);
    const std::string empty;
    const ObFTWord empty_word = make_word(empty, common::CS_TYPE_UTF8MB4_GENERAL_CI);
    output = true;
    EXPECT_EQ(common::OB_INVALID_ARGUMENT, checker.check_stopword(empty_word, output));
    EXPECT_TRUE(output);
    checker.destroy();
    output = false;
    EXPECT_EQ(common::OB_NOT_INIT, checker.check_stopword(word, output));
    EXPECT_FALSE(output);
    checker.destroy();
  }
}

} // namespace
} // namespace storage
} // namespace oceanbase

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
