/* Copyright (c) 2025 OceanBase.
 * Licensed under the Apache License, Version 2.0.
 */

#include <gtest/gtest.h>
#include <array>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include "data_plane/fts/ob_fts_parser_helper.h"

namespace oceanbase
{
namespace storage
{
using common::ObString;

static const char *const PARSER_NAMES[] = {"space", "ngram", "beng", "ik", "ngram2"};

static std::string uppercase_name(const char *name)
{
  std::string result(name);
  for (char &ch : result) {
    if (ch >= 'a' && ch <= 'z') {
      ch = static_cast<char>(ch - 'a' + 'A');
    }
  }
  return result;
}

static void check_serialized_reference(ObFTParser &parser, const char *name,
                                       const int64_t version, const int64_t capacity)
{
  SCOPED_TRACE(name);
  SCOPED_TRACE(version);
  SCOPED_TRACE(capacity);
  std::array<char, 128> actual;
  std::array<char, 128> expected;
  actual.fill('\x5a');
  expected.fill('\x5a');
  // The entire allocation, including bytes beyond the terminator and caller
  // capacity, is checked. The reference uses the independent libc formatter.
  const int length = std::snprintf(expected.data() + 1, static_cast<size_t>(capacity),
                                    "%s.%" PRId64, name, version);
  ASSERT_GT(length, 0);
  ASSERT_LT(length, capacity);
  ASSERT_EQ(common::OB_SUCCESS, parser.serialize_to_str(actual.data() + 1, capacity));
  EXPECT_EQ(0, std::memcmp(actual.data(), expected.data(), actual.size()));
  EXPECT_EQ('\0', actual[static_cast<size_t>(length) + 1]);
  ObFTParser round_trip;
  ASSERT_EQ(common::OB_SUCCESS, round_trip.parse_from_str(actual.data() + 1, length));
  EXPECT_EQ(version, round_trip.get_parser_version());
  EXPECT_STREQ(name, round_trip.get_parser_name().str());
  EXPECT_TRUE(parser == round_trip);
}

TEST(FTSParserMetadata, CanonicalNamesAndCaseMatchReference)
{
  for (const char *name : PARSER_NAMES) {
    std::string upper = uppercase_name(name);
    std::string mixed(name);
    for (size_t i = 1; i < mixed.size(); i += 2) {
      mixed[i] = upper[i];
    }
    const std::string variants[] = {std::string(name), upper, mixed};
    const ObIFTParserDesc *reference_desc = nullptr;
    ObFTParser reference_parser;
    ASSERT_EQ(common::OB_SUCCESS, reference_parser.init(ObString::make_string(name)));
    ASSERT_EQ(common::OB_SUCCESS, reference_parser.get_desc(reference_desc));
    ASSERT_NE(nullptr, reference_desc);
    for (const std::string &variant : variants) {
      SCOPED_TRACE(variant);
      ObFTParser parser;
      ASSERT_EQ(common::OB_SUCCESS, parser.init(ObString::make_string(variant.c_str())));
      ASSERT_TRUE(parser.is_builtin());
      ASSERT_TRUE(parser.is_valid());
      EXPECT_STREQ(name, parser.get_parser_name().str());
      EXPECT_EQ(0 == std::strcmp(name, "space"), parser.is_space());
      EXPECT_EQ(0 == std::strcmp(name, "ngram"), parser.is_ngram());
      EXPECT_EQ(0 == std::strcmp(name, "beng"), parser.is_beng());
      EXPECT_EQ(0 == std::strcmp(name, "ik"), parser.is_ik());
      EXPECT_EQ(0 == std::strcmp(name, "ngram2"), parser.is_ngram2());
      const ObIFTParserDesc *desc = nullptr;
      ASSERT_EQ(common::OB_SUCCESS, parser.get_desc(desc));
      EXPECT_EQ(reference_desc, desc);
      for (const int64_t capacity : {64, 65}) {
        check_serialized_reference(parser, name, ObFTParser::BUILTIN_VERSION, capacity);
      }
    }
  }
}

TEST(FTSParserMetadata, VersionBoundariesMatchReferenceAndRoundTrip)
{
  const int64_t versions[] = {0, 1, 9, 10, std::numeric_limits<int64_t>::max()};
  for (const char *name : PARSER_NAMES) {
    const std::string upper = uppercase_name(name);
    ObFTParserName canonical;
    ASSERT_EQ(common::OB_SUCCESS, canonical.set_name(upper.c_str()));
    for (const int64_t version : versions) {
      ObFTParser parser;
      parser.set_name_and_version(canonical, version);
      ASSERT_TRUE(parser.is_valid());
      for (const int64_t capacity : {64, 65}) {
        check_serialized_reference(parser, name, version, capacity);
      }
    }
  }
}

TEST(FTSParserMetadata, InvalidStateAndBuffersPreserveFailureContract)
{
  ObFTParser valid;
  ASSERT_EQ(common::OB_SUCCESS, valid.init(ObString::make_string("beng")));
  std::array<char, 128> sentinel;
  sentinel.fill('\x5a');
  for (const int64_t capacity : {-1, 0, 1, 2, 8, 63}) {
    auto actual = sentinel;
    EXPECT_EQ(common::OB_INVALID_ARGUMENT, valid.serialize_to_str(actual.data() + 1, capacity));
    EXPECT_EQ(actual, sentinel);
  }
  for (const int64_t capacity : {0, 63, 64, 65}) {
    EXPECT_EQ(common::OB_INVALID_ARGUMENT, valid.serialize_to_str(nullptr, capacity));
  }
  ObFTParserName beng;
  ObFTParserName unknown;
  ASSERT_EQ(common::OB_SUCCESS, beng.set_name("beng"));
  ASSERT_EQ(common::OB_SUCCESS, unknown.set_name("unknown"));
  ObFTParser invalid[5];
  invalid[1].set_name_and_version(beng, -1);
  invalid[2].set_name_and_version(beng, std::numeric_limits<int64_t>::min());
  invalid[3].set_name_and_version(unknown, 1);
  invalid[4].set_name_and_version(ObFTParserName(), 1);
  for (ObFTParser &parser : invalid) {
    ASSERT_FALSE(parser.is_valid());
    for (const int64_t capacity : {64, 65}) {
      auto actual = sentinel;
      EXPECT_EQ(common::OB_NOT_SUPPORTED, parser.serialize_to_str(actual.data() + 1, capacity));
      EXPECT_EQ(actual, sentinel);
    }
    auto actual = sentinel;
    EXPECT_EQ(common::OB_INVALID_ARGUMENT, parser.serialize_to_str(actual.data() + 1, 63));
    EXPECT_EQ(actual, sentinel);
    EXPECT_EQ(common::OB_INVALID_ARGUMENT, parser.serialize_to_str(nullptr, 64));
  }
}

} // namespace storage
} // namespace oceanbase

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
