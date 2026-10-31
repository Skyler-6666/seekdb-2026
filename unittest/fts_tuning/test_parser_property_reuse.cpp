/* Copyright (c) 2025 OceanBase.
 * Licensed under the Apache License, Version 2.0.
 */

#include <gtest/gtest.h>
#include "data_plane/fts/ob_fts_parser_helper.h"
#include "data_plane/fts/ob_fts_parser_property.h"

namespace oceanbase
{
namespace storage
{
using common::ObString;

TEST(FTSParserPropertyReuse, EmptyPropertiesRestoreIKDefaultAfterMaxWord)
{
  ObFTParser parser;
  ASSERT_EQ(common::OB_SUCCESS, parser.init(ObString::make_string("ik")));
  ObFTParserProperty reused;
  const ObString max_word = ObString::make_string("{\"ik_mode\":\"max_word\"}");
  ASSERT_EQ(common::OB_SUCCESS, reused.parse_for_parser_helper(parser, max_word));
  ASSERT_FALSE(reused.ik_mode_smart_);

  ASSERT_EQ(common::OB_SUCCESS, reused.parse_for_parser_helper(parser, ObString()));
  ObFTParserProperty fresh_default;
  ASSERT_EQ(common::OB_SUCCESS, fresh_default.parse_for_parser_helper(parser, ObString()));
  EXPECT_TRUE(reused.ik_mode_smart_);
  EXPECT_TRUE(reused.is_equal(fresh_default));

  // Restoring defaults must not prevent an explicit configuration on later reuse.
  ASSERT_EQ(common::OB_SUCCESS, reused.parse_for_parser_helper(parser, max_word));
  EXPECT_FALSE(reused.ik_mode_smart_);
}

TEST(FTSParserPropertyReuse, EmptyAndExplicitDefaultPropertiesAgree)
{
  ObFTParser parser;
  ASSERT_EQ(common::OB_SUCCESS, parser.init(ObString::make_string("ik")));
  ObFTParserProperty implicit_default;
  ObFTParserProperty explicit_default;
  ASSERT_EQ(common::OB_SUCCESS,
      implicit_default.parse_for_parser_helper(parser, ObString()));
  ASSERT_EQ(common::OB_SUCCESS, explicit_default.parse_for_parser_helper(
      parser, ObString::make_string("{\"ik_mode\":\"smart\"}")));
  EXPECT_TRUE(implicit_default.is_equal(explicit_default));
}

TEST(FTSParserPropertyReuse, EmptyPropertiesMatchDefaultsForEveryBuiltInParser)
{
  const char *names[] = {"ik", "space", "beng", "ngram", "ngram2"};
  const char *defaults[] = {
      "{\"ik_mode\":\"smart\"}",
      "{\"min_token_size\":3,\"max_token_size\":84}",
      "{\"min_token_size\":3,\"max_token_size\":84}",
      "{\"ngram_token_size\":2}",
      "{\"min_ngram_size\":2,\"max_ngram_size\":3}",
  };
  for (int i = 0; i < 5; ++i) {
    SCOPED_TRACE(names[i]);
    ObFTParser parser;
    ASSERT_EQ(common::OB_SUCCESS, parser.init(ObString::make_string(names[i])));
    ObFTParserProperty implicit_default;
    ObFTParserProperty explicit_default;
    ASSERT_EQ(common::OB_SUCCESS,
        implicit_default.parse_for_parser_helper(parser, ObString()));
    ASSERT_EQ(common::OB_SUCCESS,
        explicit_default.parse_for_parser_helper(parser, ObString::make_string(defaults[i])));
    EXPECT_TRUE(implicit_default.is_equal(explicit_default));
  }
}
} // namespace storage
} // namespace oceanbase

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
