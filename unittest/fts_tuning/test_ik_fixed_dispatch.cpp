/* Copyright (c) 2025 OceanBase.
 * Licensed under the Apache License, Version 2.0.
 */
#include <gtest/gtest.h>
#include <array>
#include <cstdlib>
#include <map>
#include <string>
#include <tuple>
#include <vector>

// Include the parser's dependencies before exposing only its private helpers,
// following the focused decoder/iterator test convention in this directory.
#include "lib/allocator/ob_allocator.h"
#include "storage/fts/dict/ob_ft_cache_container.h"
#include "storage/fts/dict/ob_ft_dict.h"
#include "storage/fts/dict/ob_ft_dict_def.h"
#include "storage/fts/ik/ob_ik_processor.h"
#include "data_plane/fts/ob_fts_parser.h"
#define private public
#include "storage/fts/ob_ik_ft_parser.h"
#undef private
#include "storage/fts/ik/ob_ik_arbitrator.h"
#include "storage/fts/ik/ob_ik_cjk_processor.h"
#include "storage/fts/ik/ob_ik_letter_processor.h"
#include "storage/fts/ik/ob_ik_quantifier_processor.h"
#include "storage/fts/ik/ob_ik_surrogate_processor.h"

namespace oceanbase
{
namespace storage
{
namespace
{

class FailingAllocator : public common::ObIAllocator
{
public:
  ~FailingAllocator() override
  {
    for (const auto &entry : live_) { std::free(entry.first); }
  }
  void fail_after(const int64_t successful_allocations)
  {
    allowance_ = successful_allocations;
  }
  void *alloc(const int64_t size) override
  {
    if (allowance_ == 0) { return nullptr; }
    if (allowance_ > 0) { --allowance_; }
    void *ptr = std::malloc(size);
    if (ptr != nullptr) { live_.emplace(ptr, size); }
    return ptr;
  }
  void *alloc(const int64_t size, const common::ObMemAttr &) override { return alloc(size); }
  void free(void *ptr) override
  {
    if (ptr != nullptr) {
      const auto it = live_.find(ptr);
      if (it == live_.end()) {
        ++invalid_frees_;
      } else {
        live_.erase(it);
        std::free(ptr);
      }
    }
  }
  size_t live_count() const { return live_.size(); }
  int invalid_frees() const { return invalid_frees_; }
private:
  int64_t allowance_ = -1;
  int invalid_frees_ = 0;
  std::map<void *, int64_t> live_;
};

class EmptyTestDictionary : public ObIFTDict
{
public:
  explicit EmptyTestDictionary(const bool fail_on_q = false) : fail_on_q_(fail_on_q) {}
  int init() override { return common::OB_SUCCESS; }
  int match(const common::ObString &, bool &match) const override
  {
    match = false;
    return common::OB_SUCCESS;
  }
  int match(const common::ObString &word, ObDATrieHit &hit) const override
  {
    hit.set_unmatch();
    return fail_on_q_ && word.length() == 1 && word.ptr()[0] == 'q'
        ? common::OB_ERR_UNEXPECTED : common::OB_SUCCESS;
  }
  int match_with_hit(const common::ObString &word,
      const ObDATrieHit &, ObDATrieHit &hit) const override { return match(word, hit); }
private:
  bool fail_on_q_;
};

TEST(IKFixedDispatch, EveryPartialProcessorAllocationIsReleased)
{
  // Context initialization has already succeeded when stage construction starts.
  // Failing every stage tests partial local cleanup, then the same reset followed
  // by destruction that init() uses on setup failure. Four is the success case.
  for (int stage = 0; stage <= 4; ++stage) {
    SCOPED_TRACE(stage);
    FailingAllocator allocator;
    EmptyTestDictionary dictionary;
    {
      ObIKFTParser parser(allocator, nullptr);
      parser.coll_type_ = common::CS_TYPE_UTF8MB4_BIN;
      ObFTParserParam param;
      param.cs_ = common::ObCharset::get_charset(common::CS_TYPE_UTF8MB4_BIN);
      param.allocator_ = &allocator;
      param.fulltext_ = "abc";
      param.ft_length_ = 3;
      ASSERT_EQ(common::OB_SUCCESS, parser.init_ctx(param));
      ASSERT_EQ(1U, allocator.live_count());
      parser.dict_main_ = &dictionary;
      parser.dict_quan_ = &dictionary;
      allocator.fail_after(stage);
      const int ret = parser.init_segmenter(param);
      EXPECT_EQ(stage == 4 ? common::OB_SUCCESS : common::OB_ALLOCATE_MEMORY_FAILED, ret);
      EXPECT_EQ(stage == 4 ? 5U : 1U, allocator.live_count());
      // The two test dictionaries are borrowed, unlike production-owned ones.
      parser.dict_main_ = nullptr;
      parser.dict_quan_ = nullptr;
      parser.reset();
      EXPECT_EQ(0U, allocator.live_count());
      EXPECT_EQ(nullptr, parser.ctx_);
      EXPECT_EQ(nullptr, parser.letter_processor_);
      EXPECT_EQ(nullptr, parser.quantifier_processor_);
      EXPECT_EQ(nullptr, parser.cjk_processor_);
      EXPECT_EQ(nullptr, parser.surrogate_processor_);
      parser.reset();
    }
    EXPECT_EQ(0U, allocator.live_count());
    EXPECT_EQ(0, allocator.invalid_frees());
  }
}

using Token = std::tuple<std::string, int64_t, int64_t>;

int run_stages(const std::string &text, const bool smart, const bool direct,
    const bool fail, std::vector<Token> &tokens)
{
  common::ObArenaAllocator allocator("IKFixedTest");
  EmptyTestDictionary dictionary(fail);
  ObIKFTParser parser(allocator, nullptr);
  parser.dict_main_ = &dictionary;
  parser.dict_quan_ = &dictionary;
  ObFTParserParam param;
  int ret = parser.init_segmenter(param);
  parser.dict_main_ = nullptr;
  parser.dict_quan_ = nullptr;
  if (ret == common::OB_SUCCESS) {
    ObIKLetterProcessor letter;
    ObIKQuantifierProcessor quantifier(dictionary, allocator);
    ObIKCJKProcessor cjk(dictionary, allocator);
    ObIKSurrogateProcessor surrogate;
    ObIIKProcessor *legacy[] = {&letter, &quantifier, &cjk, &surrogate};
    TokenizeContext ctx(common::CS_TYPE_UTF8MB4_BIN, allocator, text.data(), text.size(), smart);
    ret = ctx.init();
    while (ret == common::OB_SUCCESS && !ctx.iter_end()) {
      const char *ch = nullptr;
      uint8_t char_len = 0;
      ObFTCharUtil::CharType type;
      if ((ret = ctx.current_char(ch, char_len)) != common::OB_SUCCESS) {
      } else if ((ret = ctx.current_char_type(type)) != common::OB_SUCCESS) {
      } else if (direct) {
        ret = parser.process_one_char(ctx, ch, char_len, type);
      } else {
        for (auto *stage : legacy) {
          if ((ret = stage->do_process(ctx, ch, char_len, type)) != common::OB_SUCCESS) { break; }
        }
      }
      if (ret == common::OB_SUCCESS) { ret = ctx.step_next(); }
    }
    if (ret == common::OB_ITER_END) { ret = common::OB_SUCCESS; }
    if (ret == common::OB_SUCCESS) {
      ObIKArbitrator arbitrator;
      if ((ret = arbitrator.process(ctx)) == common::OB_SUCCESS
          && (ret = arbitrator.output_result(ctx)) == common::OB_SUCCESS) {
        const char *word = nullptr;
        int64_t len = 0, offset = 0, count = 0;
        while ((ret = ctx.get_next_token(word, len, offset, count)) == common::OB_SUCCESS) {
          tokens.emplace_back(std::string(word + offset, len), offset, count);
        }
        if (ret == common::OB_ITER_END) { ret = common::OB_SUCCESS; }
      }
    }
  }
  return ret;
}

TEST(IKFixedDispatch, ConcreteCallsMatchOriginalVirtualPipeline)
{
  const char *cases[] = {
    "a", "abc xyz", "12.50 37,500", "a+b@example.com 123a", "  !.?",
    "中国人民 一百二十三个 三十公里 abc１２３ 中文，标点！",
    "ＡＢＣ１２３ ひらがな カタカナ 한글 🙂𠀀", "\n\r\t abc ! xyz"
  };
  for (const bool smart : {false, true}) {
    for (const char *text : cases) {
      SCOPED_TRACE(text);
      std::vector<Token> expected, actual;
      ASSERT_EQ(common::OB_SUCCESS, run_stages(text, smart, false, false, expected));
      EXPECT_EQ(common::OB_SUCCESS, run_stages(text, smart, true, false, actual));
      EXPECT_EQ(expected, actual);
    }
  }
}

TEST(IKFixedDispatch, DictionaryFailureShortCircuitsLikeVirtualPipeline)
{
  for (const bool smart : {false, true}) {
    std::vector<Token> expected, actual;
    EXPECT_EQ(common::OB_ERR_UNEXPECTED, run_stages("abc q xyz", smart, false, true, expected));
    EXPECT_EQ(common::OB_ERR_UNEXPECTED, run_stages("abc q xyz", smart, true, true, actual));
    EXPECT_EQ(expected, actual);
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
