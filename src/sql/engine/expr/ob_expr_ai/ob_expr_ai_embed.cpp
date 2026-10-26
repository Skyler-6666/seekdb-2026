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

#define USING_LOG_PREFIX SQL_ENG
#include "ob_expr_ai_embed.h"
#include "query/ai/ob_ai_endpoint_resolver.h"
#include "share/rc/ob_server_runtime.h"

using namespace oceanbase::common;
using namespace oceanbase::sql;

namespace oceanbase 
{
namespace sql 
{
ObExprAIEmbed::ObExprAIEmbed(common::ObIAllocator &alloc)
    : ObFuncExprOperator(alloc, 
                        T_FUN_SYS_AI_EMBED, 
                        N_AI_EMBED, 
                        MORE_THAN_ZERO,
                        NOT_VALID_FOR_GENERATED_COL, 
                        NOT_ROW_DIMENSION) 
{
}

ObExprAIEmbed::~ObExprAIEmbed() 
{
}

int ObExprAIEmbed::calc_result_typeN(ObExprResType &type,
                                     ObExprResType *types_stack,
                                     int64_t param_num,
                                     common::ObExprTypeCtx &type_ctx) const 
{
  UNUSED(type_ctx);
  int ret = OB_SUCCESS;
  if (OB_UNLIKELY(param_num > 3 || param_num < 2)) {
    ObString func_name_(get_name());
    ret = OB_ERR_PARAM_SIZE;
    LOG_USER_ERROR(OB_ERR_PARAM_SIZE, func_name_.length(), func_name_.ptr());
  } else {
    types_stack[MODEL_IDX].set_calc_type(ObVarcharType);
    types_stack[MODEL_IDX].set_calc_collation_type(CS_TYPE_UTF8MB4_BIN);
    if (ob_is_text_tc(types_stack[CONTENT_IDX].get_type())) {
      // lob content is read out at eval time, keep the original type here
    } else {
    types_stack[CONTENT_IDX].set_calc_type(ObVarcharType);
    types_stack[CONTENT_IDX].set_calc_collation_type(CS_TYPE_UTF8MB4_BIN);
    }
    if (OB_FAIL(ret)) {
    } else if (param_num == 3) {
      if (ob_is_integer_type(types_stack[DIM_IDX].get_type())) {
        types_stack[DIM_IDX].set_calc_type(ObIntType);
        types_stack[DIM_IDX].set_precision(10);
        types_stack[DIM_IDX].set_scale(0);
      } else {
        ret = OB_INVALID_ARGUMENT;
        LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_embed, dimension parameter must be an integer, not a decimal or float");
      }
    }
    if (OB_SUCC(ret)) {
    type.set_varchar();
    type.set_collation_type(CS_TYPE_UTF8MB4_BIN);
    type.set_collation_level(CS_LEVEL_COERCIBLE);
  }
  }
  return ret;
}

int ObExprAIEmbed::eval_ai_embed(const ObExpr &expr, ObEvalCtx &ctx, ObDatum &res) 
{
  INIT_SUCC(ret);
  ObDatum *arg_model_id = nullptr;
  ObDatum *arg_content = nullptr;
  ObDatum *arg_dim = nullptr;
  if (expr.arg_cnt_ == 3 ? OB_FAIL(expr.eval_param_value(ctx, arg_model_id, arg_content, arg_dim))
                         : OB_FAIL(expr.eval_param_value(ctx, arg_model_id, arg_content))) {
  } else if (arg_model_id->is_null()) {
    ret = OB_INVALID_ARGUMENT;
    LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_embed, model id is null");
    res.set_null();
  } else if (arg_content->is_null()) {
    // a null input row does not call the model, the result is null as well
    res.set_null();
  } else {
    ObEvalCtx::TempAllocGuard tmp_alloc_g(ctx);
    
    MultimodeAlloctor temp_allocator(tmp_alloc_g.get_allocator());
    lib::ObMallocHookAttrGuard malloc_guard(lib::ObMemAttr(N_AI_EMBED));
    ObAIFuncExprInfo *info = nullptr;
    share::ObAiModelEndpointInfo resolved_endpoint;
    const share::ObAiModelEndpointInfo *endpoint_info = &resolved_endpoint;
    query::ObIAiEndpointResolver *endpoint_resolver =
        ::oceanbase::share::server_service<::oceanbase::query::ObIAiEndpointResolver>();
    ObString model_id = arg_model_id->get_string();
    ObString content;
    if (OB_FAIL(ObTextStringHelper::get_string(ctx.exec_ctx_, expr, temp_allocator, CONTENT_IDX,
                                               arg_content, content))) {
    } else if (model_id.empty() || content.empty()) {
      ret = OB_INVALID_ARGUMENT;
      LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_embed, model id or input is empty");
      res.set_null();
    }
    int64_t dim = 0;
    ObJsonInt *dim_json = nullptr;
    ObJsonObject *config = nullptr;
    if (OB_FAIL(ret)) {
    } else if (OB_NOT_NULL(arg_dim) && !arg_dim->is_null()) {
      dim = arg_dim->get_int();
      if (dim <= 0) {
        ret = OB_INVALID_ARGUMENT;
        LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_embed, dimension parameter must be a positive integer");
        res.set_null();
      } else if (OB_FAIL(ObAIFuncJsonUtils::get_json_object(temp_allocator, config))) {
      } else if (OB_FAIL(ObAIFuncJsonUtils::get_json_int(temp_allocator, dim, dim_json))) {
      } else if (OB_FAIL(config->add("dimensions", dim_json))) {
      }
    }
    if (OB_FAIL(ret)) {
    } else if (OB_FAIL(ObAIFuncUtils::get_ai_func_info(temp_allocator, model_id, info))) {
    } else if (OB_ISNULL(endpoint_resolver)) {
      ret = OB_ERR_UNEXPECTED;
    } else if (OB_FAIL(endpoint_resolver->resolve_by_model_name(
                   model_id, temp_allocator, resolved_endpoint))) {
    } else {
      ObAIFuncModel model(temp_allocator, *info, *endpoint_info);
      ObString result;
      if (OB_FAIL(model.call_dense_embedding(content, config, result))) {
      } else if (OB_FAIL(ObAIFuncUtils::set_string_result(expr, ctx, res, result))) {
      }
    }
  }
  return ret;
}

int ObExprAIEmbed::eval_ai_embed_batch(const ObExpr &expr, ObEvalCtx &ctx,
                                       const ObBitVector &skip, const int64_t batch_size)
{
  INIT_SUCC(ret);
  ObDatum *res_datums = expr.locate_batch_datums(ctx);
  ObBitVector &eval_flags = expr.get_evaluated_flags(ctx);
  // the inputs of the whole batch are collected first, then the rows sharing
  // the same model and the same dimension are sent in one request.
  ObArray<ObString> contents;
  ObArray<int64_t> row_idxs;
  ObArray<ObString> models;
  ObArray<int64_t> dims;
  ObArray<int64_t> has_dims;
  ObExpr *model_expr = expr.args_[MODEL_IDX];
  ObExpr *content_expr = expr.args_[CONTENT_IDX];
  ObExpr *dim_expr = expr.arg_cnt_ > DIM_IDX ? expr.args_[DIM_IDX] : nullptr;
  ObEvalCtx::TempAllocGuard tmp_alloc_g(ctx);
  MultimodeAlloctor temp_allocator(tmp_alloc_g.get_allocator());
  lib::ObMallocHookAttrGuard malloc_guard(lib::ObMemAttr(N_AI_EMBED));

  if (OB_ISNULL(res_datums) || OB_ISNULL(model_expr) || OB_ISNULL(content_expr)) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("unexpected null", K(ret), K(res_datums), K(model_expr), K(content_expr));
  } else if (OB_FAIL(model_expr->eval_batch(ctx, skip, batch_size))) {
    LOG_WARN("eval model id failed", K(ret));
  } else if (OB_FAIL(content_expr->eval_batch(ctx, skip, batch_size))) {
    LOG_WARN("eval content failed", K(ret));
  } else if (OB_NOT_NULL(dim_expr) && OB_FAIL(dim_expr->eval_batch(ctx, skip, batch_size))) {
    LOG_WARN("eval dimension failed", K(ret));
  } else {
    for (int64_t i = 0; OB_SUCC(ret) && i < batch_size; i++) {
      if (skip.at(i) || eval_flags.at(i)) {
        continue;
      }
      ObDatum &model_datum = model_expr->locate_expr_datum(ctx, i);
      ObDatum &content_datum = content_expr->locate_expr_datum(ctx, i);
      bool row_has_dim = false;
      int64_t row_dim = 0;
      if (OB_NOT_NULL(dim_expr)) {
        ObDatum &dim_datum = dim_expr->locate_expr_datum(ctx, i);
        if (!dim_datum.is_null()) {
          row_has_dim = true;
          row_dim = dim_datum.get_int();
        }
      }
      if (model_datum.is_null()) {
        ret = OB_INVALID_ARGUMENT;
        LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_embed, model id is null");
        LOG_WARN("ai_embed model id is null", K(ret), K(i));
      } else if (model_datum.get_string().empty()) {
        ret = OB_INVALID_ARGUMENT;
        LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_embed, model id or input is empty");
        LOG_WARN("ai_embed model id is empty", K(ret), K(i));
      } else if (row_has_dim && row_dim <= 0) {
        ret = OB_INVALID_ARGUMENT;
        LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_embed, dimension parameter must be a positive integer");
        LOG_WARN("ai_embed invalid dimension", K(ret), K(i), K(row_dim));
      } else if (content_datum.is_null()) {
        // a null input row does not call the model, the result is null as well
        res_datums[i].set_null();
        eval_flags.set(i);
      } else {
        ObString content;
        if (OB_FAIL(ObTextStringHelper::get_string(ctx.exec_ctx_, expr, temp_allocator, CONTENT_IDX,
                                                   &content_datum, content))) {
          LOG_WARN("read content failed", K(ret), K(i));
        } else if (content.empty()) {
          ret = OB_INVALID_ARGUMENT;
          LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_embed, model id or input is empty");
          LOG_WARN("ai_embed input is empty", K(ret), K(i));
        } else if (OB_FAIL(contents.push_back(content))) {
          LOG_WARN("push content failed", K(ret));
        } else if (OB_FAIL(row_idxs.push_back(i))) {
          LOG_WARN("push row index failed", K(ret));
        } else if (OB_FAIL(models.push_back(model_datum.get_string()))) {
          LOG_WARN("push model id failed", K(ret));
        } else if (OB_FAIL(dims.push_back(row_dim))) {
          LOG_WARN("push dimension failed", K(ret));
        } else if (OB_FAIL(has_dims.push_back(row_has_dim ? 1 : 0))) {
          LOG_WARN("push dimension flag failed", K(ret));
        }
      }
    }
  }

  if (OB_FAIL(ret)) {
  } else if (!contents.empty()) {
    // build the distinct (model, dimension) groups, every group is answered by
    // one request, so a query that uses a constant model still batches all of
    // its rows while a per row model keeps its own request.
    ObArray<ObString> group_models;
    ObArray<int64_t> group_dims;
    ObArray<int64_t> group_has_dims;
    for (int64_t r = 0; OB_SUCC(ret) && r < contents.count(); r++) {
      bool found = false;
      for (int64_t g = 0; !found && g < group_models.count(); g++) {
        if (has_dims.at(r) == group_has_dims.at(g) &&
            0 == models.at(r).compare(group_models.at(g)) &&
            (0 == has_dims.at(r) || dims.at(r) == group_dims.at(g))) {
          found = true;
        }
      }
      if (!found) {
        if (OB_FAIL(group_models.push_back(models.at(r)))) {
          LOG_WARN("push group model failed", K(ret));
        } else if (OB_FAIL(group_dims.push_back(dims.at(r)))) {
          LOG_WARN("push group dimension failed", K(ret));
        } else if (OB_FAIL(group_has_dims.push_back(has_dims.at(r)))) {
          LOG_WARN("push group dimension flag failed", K(ret));
        }
      }
    }
    for (int64_t g = 0; OB_SUCC(ret) && g < group_models.count(); g++) {
      const ObString &group_model = group_models.at(g);
      const int64_t group_dim = group_dims.at(g);
      const bool group_has_dim = group_has_dims.at(g) > 0;
      ObArray<ObString> group_contents;
      ObArray<int64_t> group_rows;
      for (int64_t r = 0; OB_SUCC(ret) && r < contents.count(); r++) {
        if (has_dims.at(r) != group_has_dims.at(g) ||
            0 != models.at(r).compare(group_model) ||
            (group_has_dim && dims.at(r) != group_dim)) {
          continue;
        }
        if (OB_FAIL(group_contents.push_back(contents.at(r)))) {
          LOG_WARN("push content failed", K(ret));
        } else if (OB_FAIL(group_rows.push_back(row_idxs.at(r)))) {
          LOG_WARN("push row index failed", K(ret));
        }
      }
      if (OB_FAIL(ret) || group_contents.empty()) {
      } else {
        ObAIFuncExprInfo *info = nullptr;
        share::ObAiModelEndpointInfo resolved_endpoint;
        query::ObIAiEndpointResolver *endpoint_resolver =
            ::oceanbase::share::server_service<::oceanbase::query::ObIAiEndpointResolver>();
        ObJsonObject *config = nullptr;
        if (group_has_dim) {
          ObJsonInt *dim_json = nullptr;
          if (OB_FAIL(ObAIFuncJsonUtils::get_json_object(temp_allocator, config))) {
          } else if (OB_FAIL(ObAIFuncJsonUtils::get_json_int(temp_allocator, group_dim, dim_json))) {
          } else if (OB_FAIL(config->add("dimensions", dim_json))) {
          }
        }
        ObArray<ObString> results;
        if (OB_FAIL(ret)) {
        } else if (OB_FAIL(ObAIFuncUtils::get_ai_func_info(temp_allocator, group_model, info))) {
        } else if (OB_ISNULL(endpoint_resolver)) {
          ret = OB_ERR_UNEXPECTED;
        } else if (OB_FAIL(endpoint_resolver->resolve_by_model_name(
                       group_model, temp_allocator, resolved_endpoint))) {
        } else {
          ObAIFuncModel model(temp_allocator, *info, resolved_endpoint);
          if (OB_FAIL(model.call_dense_embedding_vector_v2(group_contents, config, results))) {
            LOG_WARN("call dense embedding failed", K(ret), K(group_contents.count()));
          } else if (results.count() != group_contents.count()) {
            ret = OB_ERR_UNEXPECTED;
            LOG_WARN("result count mismatch", K(ret), K(results.count()), K(group_contents.count()));
          } else {
            for (int64_t i = 0; OB_SUCC(ret) && i < results.count(); i++) {
              const ObString &result = results.at(i);
              int64_t idx = group_rows.at(i);
              char *res_buf = expr.get_str_res_mem(ctx, result.length(), idx);
              if (OB_ISNULL(res_buf)) {
                ret = OB_ALLOCATE_MEMORY_FAILED;
                LOG_WARN("allocate result buffer failed", K(ret), K(result.length()));
              } else {
                MEMCPY(res_buf, result.ptr(), result.length());
                res_datums[idx].set_string(res_buf, result.length());
                eval_flags.set(idx);
              }
            }
          }
        }
      }
    }
  }
  return ret;
}

int ObExprAIEmbed::cg_expr(ObExprCGCtx &expr_cg_ctx, 
                           const ObRawExpr &raw_expr,
                           ObExpr &rt_expr) const 
{
  int ret = OB_SUCCESS;
  // TODO: support schema version match in plan cache for ai func
  // const ObRawExpr *model_key = raw_expr.get_param_expr(0);
  // if (OB_NOT_NULL(model_key)
  //     && (model_key->is_static_scalar_const_expr() || model_key->is_const_expr())
  //     && model_key->get_expr_type() != T_OP_GET_USER_VAR &&
  //     OB_NOT_NULL(expr_cg_ctx.schema_guard_)) {
  //   ObIAllocator *allocator = expr_cg_ctx.allocator_;
  //   ObExecContext *exec_ctx = expr_cg_ctx.session_->get_cur_exec_ctx();
  //   ObObj const_data;
  //   bool got_data = false;
  //   ObAIFuncExprInfo *info = nullptr;
  //   if (OB_ISNULL(allocator)) {
  //     ret = OB_ERR_UNEXPECTED;
  //     LOG_WARN("allocator is null", K(ret));
  //   } else if (OB_FAIL(ObSQLUtils::calc_const_or_calculable_expr(exec_ctx,
  //                                                         model_key,
  //                                                         const_data,
  //                                                         got_data,
  //                                                         *allocator))) {
  //     LOG_WARN("failed to calc offset expr", K(ret));
  //   } else if (!got_data || const_data.is_null()) {
  //   } else if (OB_FAIL(ObAIFuncUtils::get_ai_func_info(*allocator, const_data.get_string(), *expr_cg_ctx.schema_guard_, info))) {
  //     LOG_WARN("failed to get ai func info", K(ret), K(const_data.get_string()));
  //   } else {
  //     rt_expr.extra_info_ = info;
  //   }
  // }

  if (OB_SUCC(ret)) {
    rt_expr.eval_func_ = ObExprAIEmbed::eval_ai_embed;
    rt_expr.eval_batch_func_ = ObExprAIEmbed::eval_ai_embed_batch;
  }
  return ret;
}

} // namespace sql
} // namespace oceanbase
