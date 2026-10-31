/* Copyright (c) 2025 OceanBase.
 * Licensed under the Apache License, Version 2.0.
 */

#include <gtest/gtest.h>
#include <cstring>
#include <set>
#include <string>
#include <vector>
#include "storage/fts/dict/ob_ft_dat_dict.h"

namespace oceanbase
{
namespace storage
{
namespace
{

// Keep the original hash/probe/equality algorithm as an independent oracle.
int legacy_find(const ObArrayHashMap &map, const ObFTSingleWord &word, ObFTWordCode &code)
{
  const auto &header = *reinterpret_cast<const ObArrayHashMap::Header *>(&map);
  uint64_t index = word.get_word().hash() % header.capacity_;
  while (header.data[index].used) {
    if (header.data[index].word == word) {
      code = header.data[index].code;
      return common::OB_SUCCESS;
    }
    index = (index + 3) % header.capacity_;
  }
  return common::OB_ENTRY_NOT_EXIST;
}

class DatCharacterProbe : public testing::Test
{
protected:
  void create(const size_t word_count)
  {
    bytes_.assign(ObArrayHashMap::estimate_size(word_count), 0xcc);
    map_ = reinterpret_cast<ObArrayHashMap *>(bytes_.data());
    ASSERT_EQ(common::OB_SUCCESS, map_->init(word_count));
  }

  void compare(const std::string &text)
  {
    for (const unsigned char padding : {0, 0x5a, 0xff}) {
      ObFTSingleWord word;
      std::memset(&word, padding, sizeof(word));
      ASSERT_EQ(common::OB_SUCCESS, word.set_word(text.data(), text.size()));
      ObFTWordCode expected = -991;
      ObFTWordCode actual = -991;
      EXPECT_EQ(legacy_find(*map_, word, expected), map_->find(word, actual));
      EXPECT_EQ(expected, actual);
    }
  }

  std::vector<unsigned char> bytes_;
  ObArrayHashMap *map_ = nullptr;
};

TEST_F(DatCharacterProbe, BinaryKeysAndEverySupportedLength)
{
  static_assert(common::ObCharset::MAX_MB_LEN == 5,
      "Extend the character-length coverage when the representation changes");
  for (const size_t word_count : {0, 150, 307}) {
    create(word_count);
    std::vector<std::string> keys;
    for (int len = 1; len <= common::ObCharset::MAX_MB_LEN; ++len) {
      for (int seed = 0; seed < 6; ++seed) {
        std::string key(len, '\0');
        for (int i = 0; i < len; ++i) {
          key[i] = static_cast<char>(seed * 37 + i * 53);
        }
        ASSERT_EQ(common::OB_SUCCESS, map_->insert(common::ObString(len, key.data()), keys.size() + 1));
        keys.push_back(key);
      }
    }
    for (const auto &key : keys) {
      compare(key);
      for (int position = 0; position < key.size(); ++position) {
        std::string different = key;
        different[position] ^= 0x80;
        compare(different);
      }
    }
    for (int value = 0; value <= 255; ++value) {
      compare(std::string(1, static_cast<char>(value)));
    }
  }
}

TEST_F(DatCharacterProbe, CollisionsCrossEndAndPreserveMissOutput)
{
  for (const uint64_t target_bucket : {99, 100}) {
    create(0); // capacity 101, including a +3 probe across the last bucket.
    ASSERT_EQ(101U, map_->capacity());
    std::vector<std::string> collisions;
    for (uint64_t candidate = 1; collisions.size() < 31; ++candidate) {
      ASSERT_LT(candidate, 1000000U);
      std::string key(5, '\0');
      for (size_t i = 0; i < key.size(); ++i) {
        key[i] = static_cast<char>(candidate >> (i * 8));
      }
      if (common::ObString(key.size(), key.data()).hash() % map_->capacity() == target_bucket) {
        collisions.push_back(key);
      }
    }
    for (size_t i = 0; i + 1 < collisions.size(); ++i) {
      const auto &key = collisions[i];
      ASSERT_EQ(common::OB_SUCCESS,
          map_->insert(common::ObString(key.size(), key.data()), 1000 + i));
    }
    for (const auto &key : collisions) {
      compare(key); // Includes a miss after the entire collision chain.
    }
  }
}

TEST_F(DatCharacterProbe, EmptyKeyAndRelocatedDictionary)
{
  create(0);
  compare("");
  ASSERT_EQ(common::OB_SUCCESS, map_->insert(common::ObString(0, ""), 71));
  compare("");
  const std::string nul(1, '\0');
  ASSERT_EQ(common::OB_SUCCESS, map_->insert(common::ObString(1, nul.data()), 72));
  compare(nul);
  compare(std::string(2, '\0'));

  auto relocated = bytes_;
  map_ = reinterpret_cast<ObArrayHashMap *>(relocated.data());
  compare("");
  compare(nul);
  compare(std::string(2, '\0'));
}

} // namespace
} // namespace storage
} // namespace oceanbase

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
