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

#include "sql/engine/expr/ob_expr_tokenize.h"

#include "lib/alloc/alloc_struct.h"
#include "lib/allocator/page_arena.h"
#include "lib/charset/ob_charset.h"
#include "lib/container/ob_se_array.h"
#include "common/json_type/ob_json_base.h"
#include "common/json_type/ob_json_tree.h"
#include "lib/ob_errno.h"
#include "lib/oblog/ob_log_module.h"
#include "lib/string/ob_string.h"
#include "lib/utility/ob_macro_utils.h"
#include "object/ob_object.h"
#include "sql/resolver/ddl/ob_fts_index_builder_util.h"
#include "share/ob_json_access_utils.h"
#include "data_plane/fts/ob_fts_parser_helper.h"
#include "data_plane/fts/dict/ob_gen_dic_loader.h"
#include "data_plane/fts/ob_fts_parser_property.h"

#define USING_LOG_PREFIX SQL_ENG
#include "sql/engine/expr/ob_expr_json_func_helper.h" // file not self-contained, there're logs inside.

namespace oceanbase
{
namespace sql
{

namespace
{

struct CachedToken
{
  CachedToken() : word_(), count_(0) {}
  ObString word_;
  int64_t count_;
};

// TOKENIZE() is evaluated repeatedly with the same parser and properties by
// applications and benchmarks. Keep only parser construction state per thread;
// every call still tokenizes its complete input and builds a fresh result.
//
// While the parser is unchanged, segmenting the same text yields the same
// ordered token list, so the previous call's tokens are kept as well. A text
// hit skips segmentation but still builds a complete JSON document in the
// caller's allocator.
struct TokenizeParserCache
{
  TokenizeParserCache()
    : arena_(lib::ObMemAttr("TokenizeParse")),
      token_arena_(lib::ObMemAttr("TokenizeToken")),
      parser_name_(),
      properties_(),
      collation_(CS_TYPE_INVALID),
      helper_(),
      dic_loader_(),
      dic_checked_(false),
      valid_(false),
      text_key_(),
      doc_len_(0),
      tokens_(),
      text_valid_(false)
  {}
  void reset_text()
  {
    text_valid_ = false;
    text_key_.reset();
    doc_len_ = 0;
    tokens_.reuse();
    token_arena_.reset();
  }

  common::ObArenaAllocator arena_;
  common::ObArenaAllocator token_arena_;
  ObString parser_name_;
  ObString properties_;
  ObCollationType collation_;
  storage::ObFTParseHelper helper_;
  storage::ObDicLoaderHandle dic_loader_;
  bool dic_checked_;
  bool valid_;
  ObString text_key_;
  int64_t doc_len_;
  common::ObSEArray<CachedToken, 128> tokens_;
  bool text_valid_;
};

thread_local TokenizeParserCache g_tokenize_parser_cache;

int save_cached_tokens(TokenizeParserCache &cache,
                       const storage::ObFTWordMap &token_map,
                       const int64_t doc_len,
                       const ObString &text)
{
  int ret = OB_SUCCESS;
  for (storage::ObFTWordMap::const_iterator iter = token_map.begin();
      OB_SUCC(ret) && iter != token_map.end();
      ++iter) {
    CachedToken entry;
    if (OB_FAIL(ob_write_string(cache.token_arena_,
                                iter->first.get_word().get_string(),
                                entry.word_))) {
    } else {
      entry.count_ = iter->second;
      if (OB_FAIL(cache.tokens_.push_back(entry))) {
      }
    }
  }
  if (OB_FAIL(ret)) {
  } else if (OB_FAIL(ob_write_string(cache.token_arena_, text, cache.text_key_))) {
  } else {
    cache.doc_len_ = doc_len;
    cache.text_valid_ = true;
  }
  return ret;
}

int make_cached_token_array_json(const common::ObSEArray<CachedToken, 128> &tokens,
                                 common::ObIAllocator &allocator,
                                 common::ObIJsonBase *&json_root)
{
  int ret = OB_SUCCESS;
  ObJsonArray *token_array = nullptr;
  if (OB_ISNULL(token_array = OB_NEWx(ObJsonArray, &allocator, &allocator))) {
    ret = OB_ALLOCATE_MEMORY_FAILED;
  } else {
    for (int64_t i = 0; OB_SUCC(ret) && i < tokens.count(); ++i) {
      ObJsonString *token = nullptr;
      if (OB_ISNULL(token = OB_NEWx(ObJsonString, &allocator, tokens.at(i).word_))) {
        ret = OB_ALLOCATE_MEMORY_FAILED;
      } else if (OB_FAIL(token_array->append(token))) {
        OB_DELETEx(ObJsonString, &allocator, token);
      }
    }
  }
  if (OB_SUCC(ret)) {
    json_root = token_array;
  } else {
    OB_DELETEx(ObJsonArray, &allocator, token_array);
  }
  return ret;
}

int make_cached_detail_json(const common::ObSEArray<CachedToken, 128> &tokens,
                            const int64_t doc_len,
                            common::ObIAllocator &allocator,
                            common::ObIJsonBase *&json_root)
{
  int ret = OB_SUCCESS;
  ObJsonObject *root_obj = nullptr;
  ObJsonInt *cnt = nullptr;
  ObJsonArray *token_array = nullptr;
  if (OB_ISNULL(root_obj = OB_NEWx(ObJsonObject, &allocator, &allocator))) {
    ret = OB_ALLOCATE_MEMORY_FAILED;
  } else if (OB_ISNULL(cnt = OB_NEWx(ObJsonInt, &allocator, doc_len))) {
    ret = OB_ALLOCATE_MEMORY_FAILED;
  } else if (OB_ISNULL(token_array = OB_NEWx(ObJsonArray, &allocator, &allocator))) {
    ret = OB_ALLOCATE_MEMORY_FAILED;
  } else {
    for (int64_t i = 0; OB_SUCC(ret) && i < tokens.count(); ++i) {
      ObJsonObject *node = nullptr;
      ObJsonInt *token_cnt = nullptr;
      if (OB_ISNULL(node = OB_NEWx(ObJsonObject, &allocator, &allocator))) {
        ret = OB_ALLOCATE_MEMORY_FAILED;
      } else if (OB_ISNULL(token_cnt = OB_NEWx(ObJsonInt, &allocator, tokens.at(i).count_))) {
        ret = OB_ALLOCATE_MEMORY_FAILED;
      } else if (OB_FAIL(node->add(tokens.at(i).word_, token_cnt))) {
      } else if (OB_FAIL(token_array->append(node))) {
      }
      if (OB_FAIL(ret)) {
        OB_DELETEx(ObJsonObject, &allocator, node);
        OB_DELETEx(ObJsonInt, &allocator, token_cnt);
      }
    }
    if (OB_SUCC(ret)) {
      if (OB_FAIL(root_obj->add("tokens", token_array))) {
      } else if (OB_FAIL(root_obj->add("doc_len", cnt))) {
      }
    }
  }
  if (OB_SUCC(ret)) {
    json_root = root_obj;
  } else {
    OB_DELETEx(ObJsonObject, &allocator, root_obj);
    OB_DELETEx(ObJsonInt, &allocator, cnt);
    OB_DELETEx(ObJsonArray, &allocator, token_array);
  }
  return ret;
}

} // anonymous namespace

ObExprTokenize::ObExprTokenize(common::ObIAllocator &alloc)
    : ObStringExprOperator(alloc,
                           T_FUN_TOKENIZE,
                           N_TOKENIZE,
                           MORE_THAN_ZERO,
                           VALID_FOR_GENERATED_COL)
{
}

ObExprTokenize::~ObExprTokenize() {}

int ObExprTokenize::eval_tokenize(const ObExpr &expr, ObEvalCtx &ctx, ObDatum &expr_datum)
{
  int ret = OB_SUCCESS;

  ObEvalCtx::TempAllocGuard tmp_alloc_g(ctx);
  common::ObArenaAllocator &temp_allocator = tmp_alloc_g.get_allocator();

  ObIJsonBase *json_result = nullptr;
  TokenizeParam param;

  // check param num, which is checked in ObExprOperator::calc_result_typeN.
  if (OB_UNLIKELY(expr.arg_cnt_ < 1 || expr.arg_cnt_ > 3)) {
    ret = OB_INVALID_ARGUMENT;
  } else if (OB_FAIL(parse_param(expr, ctx, temp_allocator, param))) {
  } else if (OB_FAIL(tokenize_fulltext(param, param.output_mode_, temp_allocator, json_result))) {
  } else if (OB_FAIL(ObJsonExprHelper::pack_json_res(expr,
                                                     ctx,
                                                     temp_allocator,
                                                     json_result,
                                                     expr_datum))) {
  }

  return ret;
}

int ObExprTokenize::tokenize_fulltext(const TokenizeParam &param,
                                      TokenizeParam::OUTPUT_MODE mode,
                                      ObIAllocator &allocator,
                                      ObIJsonBase *&result)
{
  int ret = OB_SUCCESS;
  TokenizeParserCache &cache = g_tokenize_parser_cache;
  const int64_t ft_word_bkt_cnt = MIN(MAX(param.fulltext_.length() / 2, 2), 997);
  int64_t doc_len = 0;
  storage::ObFTWordMap token_map;

  if (TokenizeParam::OUTPUT_MODE::DEFAULT != mode && TokenizeParam::OUTPUT_MODE::ALL != mode) {
    ret = OB_INVALID_ARGUMENT;
  } else if (!cache.valid_ || cache.parser_name_ != param.parser_name_
      || cache.properties_ != param.properties_
      || cache.collation_ != param.meta_.get_collation_type()) {
    cache.valid_ = false;
    cache.dic_loader_.reset();
    cache.dic_checked_ = false;
    cache.helper_.reset();
    cache.arena_.reset();
    cache.reset_text();
    char *name_buf = nullptr;
    char *properties_buf = nullptr;
    if (!param.parser_name_.empty()
        && OB_ISNULL(name_buf = static_cast<char *>(
            cache.arena_.alloc(param.parser_name_.length())))) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
    } else if (!param.properties_.empty()
        && OB_ISNULL(properties_buf = static_cast<char *>(
            cache.arena_.alloc(param.properties_.length())))) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
    } else {
      if (!param.parser_name_.empty()) {
        MEMCPY(name_buf, param.parser_name_.ptr(), param.parser_name_.length());
      }
      if (!param.properties_.empty()) {
        MEMCPY(properties_buf, param.properties_.ptr(), param.properties_.length());
      }
      if (param.parser_name_.empty()) {
        cache.parser_name_.reset();
      } else {
        cache.parser_name_.assign_ptr(name_buf, param.parser_name_.length());
      }
      if (param.properties_.empty()) {
        cache.properties_.reset();
      } else {
        cache.properties_.assign_ptr(properties_buf, param.properties_.length());
      }
      if (OB_FAIL(cache.helper_.init(
              &allocator, param.parser_name_, param.properties_))) {
      } else {
        cache.collation_ = param.meta_.get_collation_type();
        cache.valid_ = true;
      }
    }
  } else {
    cache.helper_.set_allocator(&allocator);
  }
  if (OB_FAIL(ret)) {
  } else if (!cache.dic_checked_) {
    bool need_to_load_dic = false;
    if (OB_FAIL(share::ObFtsIndexBuilderUtil::check_need_to_load_dic(
            param.parser_name_, need_to_load_dic))) {
    } else if (need_to_load_dic) {
      if (OB_FAIL(storage::ObGenDicLoader::get_instance().get_dic_loader(
              ObString::make_string(storage::ObFTSLiteral::PARSER_NAME_IK),
              ObCharset::charset_type_by_coll(param.meta_.get_collation_type()),
              cache.dic_loader_))) {
      } else if (OB_UNLIKELY(!cache.dic_loader_.is_valid())) {
        ret = OB_ERR_UNEXPECTED;
      } else if (OB_FAIL(cache.dic_loader_.get_loader()->try_load_dictionary_in_trans())) {
      } else {
        cache.dic_checked_ = true;
      }
    } else {
      cache.dic_checked_ = true;
    }
  }
  if (OB_FAIL(ret)) {
  } else if (cache.text_valid_ && cache.text_key_ == param.fulltext_) {
    if (TokenizeParam::OUTPUT_MODE::DEFAULT == param.output_mode_) {
      ret = make_cached_token_array_json(cache.tokens_, allocator, result);
    } else {
      ret = make_cached_detail_json(cache.tokens_, cache.doc_len_, allocator, result);
    }
  } else {
    cache.reset_text();
    if (OB_FAIL(token_map.create(ft_word_bkt_cnt, common::ObMemAttr("FTWordMap")))) {
    } else if (
        (0 != param.fulltext_.length())
        && OB_FAIL(cache.helper_.segment(
                       param.meta_,
                       param.fulltext_.ptr(),
                       param.fulltext_.length(),
                       doc_len,
                       token_map))) {
    } else {
      switch (param.output_mode_) {
      case TokenizeParam::OUTPUT_MODE::DEFAULT: {
        if (OB_FAIL(cache.helper_.make_token_array_json(token_map, result))) {
        } else {
          // pass
        }
        break;
      }
      case TokenizeParam::OUTPUT_MODE::ALL: {
        if (OB_FAIL(cache.helper_.make_detail_json(token_map, doc_len, result))) {
        } else {
          // pass
        }
        break;
      }
      default:
        ret = OB_INVALID_ARGUMENT;
      }
      if (OB_SUCC(ret) && OB_FAIL(save_cached_tokens(
                                      cache, token_map, doc_len, param.fulltext_))) {
        // the result is already complete, caching is only an optimization
        cache.reset_text();
        ret = OB_SUCCESS;
      }
    }
  }
  return ret;
}

ObExprTokenize::TokenizeParam ::TokenizeParam()
  : allocator_(ObMemAttr("TokenizeParam")),
    parser_name_buf_{},
    parser_name_(ObString(OB_DEFAULT_FULLTEXT_PARSER_NAME)),
    meta_(),
    fulltext_(),
    output_mode_(OUTPUT_MODE::DEFAULT)
{
}

int ObExprTokenize::TokenizeParam::parse_json_param(const ObIJsonBase *obj)
{
  int ret = OB_SUCCESS;
  ObString str;
  ObIJsonBase *val;

  if (OB_UNLIKELY(nullptr == obj)) {
    ret = OB_INVALID_ARGUMENT;
  } else if (ObJsonNodeType::J_OBJECT != obj->json_type()) {
    ret = OB_INVALID_ARGUMENT;
  } else if (obj->element_count() == 0) {
    // no data
  } else if (OB_FAIL(obj->get_object_value(0, str, val))) {
  } else if (0 == str.case_compare(CASE_INDICATOR_STR)) {
    if (ObJsonNodeType::J_STRING != val->json_type()) {
      ret = OB_INVALID_ARGUMENT;
    } else if (0 == ObString(val->get_data_length(), val->get_data()).case_compare("UPPER")) {
    } else if (0 == ObString(val->get_data_length(), val->get_data()).case_compare("LOWER")) {
    } else {
      ret = OB_INVALID_ARGUMENT;
    }
    ret = OB_NOT_SUPPORTED;
    LOG_USER_ERROR(OB_NOT_SUPPORTED, "case indentifier");
  } else if (0 == str.case_compare(OUTPUT_MODE_STR)) {
    if (ObJsonNodeType::J_STRING != val->json_type()) {
      ret = OB_INVALID_ARGUMENT;
      LOG_USER_ERROR(OB_INVALID_ARGUMENT, "output mode should be string default or all");
    } else if (0 == ObString(val->get_data_length(), val->get_data()).case_compare("DEFAULT")) {
      output_mode_ = DEFAULT;
    } else if (0 == ObString(val->get_data_length(), val->get_data()).case_compare("ALL")) {
      output_mode_ = ALL;
    } else {
      ret = OB_INVALID_ARGUMENT;
      LOG_USER_ERROR(OB_INVALID_ARGUMENT, "output mode should be string default or all");
    }
  } else if (0 == str.case_compare(STOPWORDS_LIST_STR)) {
    ret = OB_NOT_SUPPORTED;
    LOG_USER_ERROR(OB_NOT_SUPPORTED, "stopwords");
  } else if (0 == str.case_compare(ADDITIONAL_ARGS_STR)) {
    if (ObJsonNodeType::J_ARRAY != val->json_type()) {
      ret = OB_INVALID_ARGUMENT;
      LOG_USER_ERROR(OB_INVALID_ARGUMENT, "parser arguments");
    } else {
      ObString json_str;
      if (OB_FAIL(storage::ObFTParserJsonProps::tokenize_array_to_props_json(allocator_, val, json_str))) {
        ObSqlString message;
        message.append_fmt("format in %s form", ADDITIONAL_ARGS_STR);
        LOG_USER_ERROR(OB_INVALID_ARGUMENT, message.ptr());
      } else {
        properties_ = json_str;
      }
    }
  } else {
    ret = OB_INVALID_ARGUMENT;
    ObSqlString message;
    message.append_fmt("config: %s", str.ptr());
    LOG_USER_ERROR(OB_INVALID_ARGUMENT, message.ptr());
  }
  return ret;
}

int ObExprTokenize::parse_param(const ObExpr &expr,
                                ObEvalCtx &ctx,
                                common::ObArenaAllocator &allocator,
                                TokenizeParam &param)
{
  int ret = OB_SUCCESS;

  ObDatum *parser_params_datum;
  ObString raw_parser_name = ObString::make_string(OB_DEFAULT_FULLTEXT_PARSER_NAME);

  ObEvalCtx::TempAllocGuard tmp_alloc_g(ctx);
  
  MultimodeAlloctor temp_allocator(tmp_alloc_g.get_allocator());

  if (OB_UNLIKELY(expr.arg_cnt_ < 1 || expr.arg_cnt_ > 3)) {
    ret = OB_INVALID_ARGUMENT;
  } else if (OB_FAIL(parse_fulltext(expr, ctx, param))) {
  } else if (OB_FAIL(parse_parser_name(expr, ctx, param))) {
  } else if (OB_FAIL(parse_parser_properties(expr, ctx, temp_allocator, param))) {
  } else if (!param.properties_.empty()
             && OB_FAIL(param.reform_parser_properties(param.properties_))) {
    // Explicit user properties keep the canonicalizing JSON round-trip; when there
    // are none the parser helper defaults are used directly (see
    // ObFTParserProperty::parse_for_parser_helper empty-json handling).
  }
  return ret;
}

int ObExprTokenize::construct_ft_parser_inner_name(const ObString &input_str, TokenizeParam &param)
{
  int ret = OB_SUCCESS;
  storage::ObFTParser parser;

  if (OB_FAIL(parser.init(input_str))) {
  } else if (OB_FAIL(parser.serialize_to_str(param.parser_name_buf_, sizeof(param.parser_name_buf_)))) {
  } else {
    param.parser_name_ = ObString::make_string(param.parser_name_buf_);
  }
  return ret;
}

int ObExprTokenize::calc_result_typeN(ObExprResType &type,
                                      ObExprResType *types,
                                      int64_t param_num,
                                      common::ObExprTypeCtx &type_ctx) const
{
  int ret = OB_SUCCESS;

  if (OB_UNLIKELY(param_num < 1 || param_num > 3)) {
    ret = OB_ERR_PARAM_SIZE;
    ObString expr_name(N_TOKENIZE);
    LOG_USER_ERROR(OB_ERR_PARAM_SIZE, expr_name.length(), expr_name.ptr());
  } else {
    // just okay
  }

  ObLength length = ObAccuracy::DDL_DEFAULT_ACCURACY[ObJsonType].get_length();

  if (OB_SUCC(ret)) {
    // set res type
    type.set_json();
    type.set_length(length); // keep consistent with other json expr, maybe calc it later.

    // param type set, skip charset after first param
    for (int64_t i = 1; OB_SUCC(ret) && i < param_num; ++i) {
      if (ob_is_string_type(types[i].get_type())) {
        if (types[i].get_charset_type() != CHARSET_UTF8MB4) {
          types[i].set_calc_collation_type(CS_TYPE_UTF8MB4_BIN);
        }
      }
    }

    // handle param
    if (param_num >= 2) {
      types[1].set_varchar();
    }

    if (param_num >= 3) {
      if (OB_FAIL(ObJsonExprHelper::is_valid_for_json(types, 2, N_TOKENIZE))) {
      }
    }
  }

  return ret;
}

int ObExprTokenize::cg_expr(ObExprCGCtx &op_cg_ctx,
                            const ObRawExpr &raw_expr,
                            ObExpr &rt_expr) const
{
  int ret = OB_SUCCESS;
  CK((rt_expr.arg_cnt_ >= 1 && rt_expr.arg_cnt_ <= 3));
  if (OB_SUCC(ret)) {
    // do register
    rt_expr.eval_func_ = eval_tokenize;
  }
  return ret;
}

int ObExprTokenize::parse_fulltext(const ObExpr &expr, ObEvalCtx &ctx, TokenizeParam &param)
{
  int ret = OB_SUCCESS;

  ObDatum *fulltext_datum;

  if (OB_FAIL(expr.args_[0]->eval(ctx, fulltext_datum))) {
  } else {
    if (fulltext_datum->is_null()) {
      // do nothing, return empty result
      param.fulltext_ = ObString::make_empty_string();
    } else {
      param.fulltext_ = fulltext_datum->get_string();
    }
    param.meta_.set_varchar(); // as we hardcoded in fts_index
    param.meta_.set_collation_type(expr.args_[0]->obj_meta_.get_collation_type());
  }
  return ret;
}

int ObExprTokenize::parse_parser_name(const ObExpr &expr, ObEvalCtx &ctx, TokenizeParam &param)
{
  int ret = OB_SUCCESS;
  ObDatum *parser_datum = nullptr;
  ObString raw_parser_name;

  if (expr.arg_cnt_ < 2) {
    raw_parser_name = ObString::make_string(OB_DEFAULT_FULLTEXT_PARSER_NAME);
  } else if (OB_FAIL(expr.args_[1]->eval(ctx, parser_datum))) {
  } else {
    if (parser_datum->is_null()) {
      raw_parser_name = ObString::make_string(OB_DEFAULT_FULLTEXT_PARSER_NAME);
    } else {
      ObString name = parser_datum->get_string();
      raw_parser_name = name.trim();
    }
  }

  if (OB_FAIL(ret)) {
    // already logged
  } else if (OB_FAIL(construct_ft_parser_inner_name(raw_parser_name, param))) {
  }

  return ret;
}

int ObExprTokenize::parse_parser_properties(const ObExpr &expr,
                                            ObEvalCtx &ctx,
                                            MultimodeAlloctor &mm_alloc,
                                            TokenizeParam &param)
{
  int ret = OB_SUCCESS;
  ObIJsonBase *base = nullptr;

  if (expr.arg_cnt_ < 3) {
    // do nothing
  } else {
    bool is_null = false;
    if (OB_FAIL(ObJsonExprHelper::get_json_doc(expr, ctx, mm_alloc, 2, base, is_null))) {
    } else {
      if (ObJsonNodeType::J_ARRAY != base->json_type()) {
        ret = OB_INVALID_ARGUMENT;
        LOG_USER_ERROR(OB_INVALID_ARGUMENT, "parser args should be in an array.");
      } else {
        for (uint64_t i = 0; OB_SUCC(ret) && i < base->element_count(); ++i) {
          ObIJsonBase *node = nullptr;
          if (OB_FAIL(base->get_array_element(i, node))) {
          } else if (ObJsonNodeType::J_OBJECT != (node->json_type())) {
            ret = OB_INVALID_ARGUMENT;
          } else if (OB_FAIL(param.parse_json_param(node))) {
          }
        } // for
      }
    }
  }

  return ret;
}

int ObExprTokenize::TokenizeParam::reform_parser_properties(const ObString &properties)
{
  int ret = OB_SUCCESS;
  storage::ObFTParserJsonProps parser_properties;

  if (OB_FAIL(parser_properties.init())) {
  } else if (OB_FAIL(parser_properties.parse_from_valid_str(properties))) {
    LOG_USER_ERROR(OB_INVALID_ARGUMENT, "parser properties invalid.");
  } else if (OB_FAIL(parser_properties.rebuild_props_for_ddl(parser_name_,
                                                             ObCollationType::CS_TYPE_UTF8MB4_BIN,
                                                             true))) {
  } else if (OB_FAIL(parser_properties.to_format_json(allocator_, properties_))) {
  }

  return ret;
}

int ObExprTokenize::TokenizeParam::try_load_dictionary_for_ik()
{
  int ret = OB_SUCCESS;
  bool need_to_load_dic = false;
  storage::ObDicLoaderHandle dic_loader_handle;
  if (OB_FAIL(share::ObFtsIndexBuilderUtil::check_need_to_load_dic(
          parser_name_, need_to_load_dic))) {
  } else if (need_to_load_dic) {
    if (OB_FAIL(storage::ObGenDicLoader::get_instance().get_dic_loader(
                    ObString::make_string(storage::ObFTSLiteral::PARSER_NAME_IK), // currently only ik, use parser_name_ without version suffix
                    ObCharset::charset_type_by_coll(meta_.get_collation_type()),
                    dic_loader_handle))) {
    } else if (OB_UNLIKELY(!dic_loader_handle.is_valid())) {
      ret = OB_ERR_UNEXPECTED;
    } else if (OB_FAIL(dic_loader_handle.get_loader()->try_load_dictionary_in_trans())) {
    }
  }
  return ret;
}

} // namespace sql
} // namespace oceanbase
