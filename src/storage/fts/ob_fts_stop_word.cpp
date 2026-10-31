/*
 * Copyright (c) 2025 OceanBase.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "object/ob_object.h"
#define USING_LOG_PREFIX STORAGE_FTS

#include "share/rc/ob_server_runtime.h"
#include "storage/fts/ob_fts_stop_word.h"
#include "storage/fts/ob_fts_parser_helper.h"
#include "storage/fts/ob_fts_parser_property.h"
#include <algorithm>

namespace oceanbase
{
namespace storage
{

namespace {

// Nonzero base-32 digits make the encoding injective across both spelling and
// length. A uint64 holds up to 12 letters; the current list allows at most 9.
bool encode_ascii_stopword_key(const ObString &word, uint64_t &key)
{
  bool supported = word.length() > 0 && word.length() < FTS_STOP_WORD_MAX_LENGTH
      && word.length() <= 12 && nullptr != word.ptr();
  key = 0;
  for (int64_t i = 0; supported && i < word.length(); ++i) {
    const unsigned char lower = static_cast<unsigned char>(word.ptr()[i]) | 0x20;
    if (lower < 'a' || lower > 'z') {
      supported = false;
    } else {
      key = (key << 5) | static_cast<uint64_t>(lower - 'a' + 1);
    }
  }
  return supported;
}

struct AsciiStopWordIndex
{
  static constexpr int64_t WORD_COUNT = sizeof(ob_stop_word_list) / sizeof(ob_stop_word_list[0]);
  uint64_t keys_[WORD_COUNT] = {};
  bool ready_ = true;

  AsciiStopWordIndex()
  {
    for (int64_t i = 0; ready_ && i < WORD_COUNT; ++i) {
      ready_ = encode_ascii_stopword_key(ObString::make_string(ob_stop_word_list[i]), keys_[i]);
    }
    // If the original list ever contains non-ASCII or unsupported words, use
    // the original collation-aware set for every input, including non-matches.
    if (ready_) {
      std::sort(keys_, keys_ + WORD_COUNT);
    }
  }
};

bool try_ascii_stopword_lookup(const ObFTWord &word, bool &is_stopword)
{
  uint64_t key = 0;
  const ObCollationType type = word.get_collation_type();
  bool handled = (CS_TYPE_UTF8MB4_GENERAL_CI == type || CS_TYPE_UTF8MB4_BIN == type)
      && encode_ascii_stopword_key(word.get_word().get_string(), key);
  if (handled) {
    // This is immutable dictionary metadata, independent of tenant/input and
    // of the checker lifecycle. C++ initializes it once, without heap storage.
    static const AsciiStopWordIndex index;
    handled = index.ready_;
    if (handled) {
      is_stopword = std::binary_search(index.keys_,
          index.keys_ + AsciiStopWordIndex::WORD_COUNT, key);
    }
  }
  return handled;
}

} // namespace

////////////////////////////////////////////////////////////////////////////////
// class ObStopWordChecker
ObStopWordChecker::~ObStopWordChecker()
{
  destroy();
}

int ObStopWordChecker::init()
{
  int ret = OB_SUCCESS;
  
  

  if (inited_) {
    ret = OB_INIT_TWICE;
  } else if (OB_FAIL(stopword_set_.create(DEFAULT_STOPWORD_BUCKET_NUM, "StopWordSet", "StopWordSet"))) {
  } else {
    ObObjMeta stop_meta;
    stop_meta.set_varchar();
    stop_meta.set_collation_type(ObCollationType::CS_TYPE_UTF8MB4_GENERAL_CI);

    stopword_type_.set_meta(stop_meta);
    const int64_t stopword_count = sizeof(ob_stop_word_list) / sizeof(ob_stop_word_list[0]);
    for (int64_t i = 0; OB_SUCC(ret) && i < stopword_count; ++i) {
      ObFTWord stopword(STRLEN(ob_stop_word_list[i]), ob_stop_word_list[i], stopword_type_);
      if (OB_FAIL(stopword_set_.set_refactored(stopword))) {
      }
    }

    if (OB_SUCC(ret)) {
      inited_ = true;
    }
  }
  return ret;
}

void ObStopWordChecker::destroy()
{
  if (inited_) {
    stopword_set_.destroy();
    inited_ = false;
  }
}

int ObStopWordChecker::check_stopword(const ObFTWord &word, bool &is_stopword)
{
  int ret = OB_SUCCESS;
  
  
  if (OB_UNLIKELY(!inited_)) {
    ret = OB_NOT_INIT;
  } else if (OB_UNLIKELY(word.empty())) {
    ret = OB_INVALID_ARGUMENT;
  } else if (try_ascii_stopword_lookup(word, is_stopword)) {
    // Destination comparison is GENERAL_CI even for UTF8MB4_BIN input. For
    // pure ASCII letters its equivalence is exactly ASCII case folding.
  } else {
    const common::ObString &cmp_str = word.get_word().get_string();
    const ObCharsetType word_charset_type =
        ObCharset::charset_type_by_coll(word.get_collation_type());
    ObFTWord converted;
    if (word_charset_type == stopword_charset_type_) {
      // Both collations use the same character set. The subsequent map lookup
      // still compares under stopword_type_'s collation, so transcoding would
      // only allocate and copy identical bytes.
      converted = ObFTWord(cmp_str.length(), cmp_str.ptr(), stopword_type_);
    } else {
      common::ObArenaAllocator allocator(lib::ObMemAttr("ChkStopWord"));
      common::ObString converted_str;
      if (OB_FAIL(common::ObCharset::charset_convert(
                                         allocator,
                                         cmp_str,
                                         word.get_collation_type(),
                                         stopword_type_.get_collation_type(),
                                         converted_str))) {
      } else {
        converted = ObFTWord(converted_str.length(), converted_str.ptr(), stopword_type_);
      }
    }
    if (OB_SUCC(ret)) {
      ret = stopword_set_.exist_refactored(converted);
      if (OB_HASH_NOT_EXIST == ret) {
        is_stopword = false;
        ret = OB_SUCCESS;
      } else if (OB_HASH_EXIST == ret) {
        is_stopword = true;
        ret = OB_SUCCESS;
      } else if (OB_SUCC(ret)) {
        ret = OB_ERR_UNEXPECTED;
      }
    }
  }
  return ret;
}

////////////////////////////////////////////////////////////////////////////////
// class ObAddWord
ObAddWord::ObAddWord(
    const ObFTParserProperty &property,
    const ObObjMeta &meta,
    const ObAddWordFlag &flag,
    common::ObIAllocator &allocator,
    ObFTWordMap &word_map)
  : word_meta_(meta),
    allocator_(allocator),
    word_map_(word_map),
    min_max_word_cnt_(0),
    non_stopword_cnt_(0),
    stopword_cnt_(0),
    min_token_size_(property.min_token_size_),
    max_token_size_(property.max_token_size_),
    flag_(flag),
    casedown_cs_(nullptr),
    casedown_caseinfo_(nullptr)
{
  if (flag_.casedown()
      && CHARSET_UTF8MB4 == ObCharset::charset_type_by_coll(
          word_meta_.get_collation_type())) {
    casedown_cs_ = ObCharset::get_charset(word_meta_.get_collation_type());
    casedown_caseinfo_ = OB_NOT_NULL(casedown_cs_) ? casedown_cs_->caseinfo : nullptr;
  }
}

int ObAddWord::process_word(
    const char *word,
    const int64_t word_len,
    const int64_t char_cnt,
    const int64_t word_freq)
{
  int ret = OB_SUCCESS;
  bool is_stopword = false;
  ObFTWord src_word(word_len, word, word_meta_);
  ObFTWord dst_word;
  if (OB_ISNULL(word) || OB_UNLIKELY(0 >= word_len || 0 >= char_cnt || 0 >= word_freq)) {
    ret = OB_INVALID_ARGUMENT;
  } else if (is_min_max_word(char_cnt)) {
    ++min_max_word_cnt_;
  } else if (OB_FAIL(casedown_word(src_word, dst_word))) {
  } else if (OB_FAIL(check_stopword(dst_word, is_stopword))) {
  } else if (OB_UNLIKELY(is_stopword)) {
    ++stopword_cnt_;
  } else if (OB_FAIL(groupby_word(dst_word, word_freq))) {
  } else {
    non_stopword_cnt_ += word_freq;
  }
  return ret;
}

namespace {

// Returns true when ObCharset::tolower() on `word` is a byte-identical no-op.
// The parser setup supplies the same charset and case table used by
// ob_casedn_utf8mb4(); cache those pointers outside the per-token loop.
bool is_casedown_identity(const ObCharsetInfo *cs,
                          const ObUnicaseInfo *caseinfo,
                          const ObString &word)
{
  bool identity = !word.empty() && nullptr != word.ptr()
      && nullptr != cs && nullptr != cs->cset && nullptr != caseinfo
      && 1 == cs->casedn_multiply;
  if (identity) {
    const unsigned char *cur = reinterpret_cast<const unsigned char *>(word.ptr());
    const unsigned char *end = cur + word.length();
    while (identity && cur < end) {
      const unsigned char lead = *cur;
      if (lead < 0x80) {
        // ASCII only changes for uppercase letters.
        identity = lead < 'A' || lead > 'Z';
        ++cur;
      } else if (lead >= 0xE3 && lead <= 0xE9
                 && end - cur >= 3
                 && 0x80 == (cur[1] & 0xC0)
                 && 0x80 == (cur[2] & 0xC0)) {
        // Valid U+3000..U+9FFF sequences (CJK punctuation, kana, Hangul
        // compatibility, CJK and Yi) have no lowercase mapping.
        cur += 3;
      } else {
        ob_wc_t wc = 0;
        const int wc_len = cs->cset->mb_wc(cs, &wc, cur, end);
        if (wc_len <= 0) {
          // Invalid/incomplete sequence: keep the original tolower() behavior.
          identity = false;
        } else {
          if (wc <= caseinfo->maxchar) {
            const ObUnicaseInfoChar *page = caseinfo->page[wc >> 8];
            if (OB_NOT_NULL(page) && wc != static_cast<ob_wc_t>(page[wc & 0xFF].tolower)) {
              identity = false;
            }
          }
          if (identity) {
            cur += wc_len;
          }
        }
      }
    }
  }
  return identity;
}

} // anonymous namespace

bool ObAddWord::is_min_max_word(const int64_t c_len) const
{
  return flag_.min_max_word() && (c_len < min_token_size_ || c_len > max_token_size_);
}

int ObAddWord::casedown_word(const ObFTWord &src, ObFTWord &dst)
{
  int ret = OB_SUCCESS;
  if (OB_UNLIKELY(src.empty())) {
    ret = OB_INVALID_ARGUMENT;
  } else if (flag_.casedown()) {
    // FTS hot path: most tokens (Chinese, digits, already lower-case ASCII) are not
    // changed by casedown. Reference the parser-owned token memory directly instead
    // of allocating + copying + re-encoding a byte-identical string. That memory is
    // valid for the whole segment for every built-in parser (IK points into the
    // fulltext buffer, BENG copies into the segment allocator, space/ngram point into
    // the source buffer), which matches the lifetime ObFTWordMap keys need.
    // Non-identity input falls back to the original copy + convert path.
    const ObString &src_str = src.get_word().get_string();
    if (is_casedown_identity(casedown_cs_, casedown_caseinfo_, src_str)) {
      ObFTWord tmp(src_str.length(), src_str.ptr(), word_meta_);
      dst = tmp;
    } else {
      ObString dst_str;
      if (OB_FAIL(ObCharset::tolower(
                      word_meta_.get_collation_type(),
                      src_str,
                      dst_str,
                      allocator_))) {
      } else {
        ObFTWord tmp(dst_str.length(), dst_str.ptr(), word_meta_);
        dst = tmp;
      }
    }
  } else {
    dst = src;
  }
  return ret;
}

int ObAddWord::check_stopword(const ObFTWord &ft_word, bool &is_stopword)
{
  int ret = OB_SUCCESS;
  if (OB_UNLIKELY(ft_word.empty())) {
    ret = OB_INVALID_ARGUMENT;
  } else if (flag_.stopword()) {
    ObStopWordChecker *stop_word_checker = ObFTParseData::instance().stop_word_checker();
    if (OB_ISNULL(stop_word_checker)) {
      ret = OB_ERR_UNEXPECTED;
    } else if (OB_FAIL(stop_word_checker->check_stopword(ft_word, is_stopword))) {
    }
  }
  return ret;
}

int ObAddWord::groupby_word(const ObFTWord &word, const int64_t word_freq)
{
  int ret = OB_SUCCESS;
  if (OB_UNLIKELY(word.empty() || word_freq <= 0)) {
    ret = OB_INVALID_ARGUMENT;
  } else if (!flag_.groupby_word()) {
    if (OB_FAIL(word_map_.set_refactored(word, 1/*word count*/))) {
    }
  } else {
    // Keep the latest spelling, as set_refactored(..., overwrite) did for
    // collation-equivalent words, without replacing the existing hash node.
    auto add_word_freq = [&word, word_freq](common::hash::HashMapPair<ObFTWord, int64_t> &entry) {
      entry.first = word;
      entry.second += word_freq;
    };
    if (OB_FAIL(word_map_.set_or_update(word, 1/*word count*/, add_word_freq))) {
    }
  }
  return ret;
}

} // end namespace storage
} // end namespace oceanbase
