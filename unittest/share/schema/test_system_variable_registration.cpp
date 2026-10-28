/* Copyright (c) 2026 OceanBase.
 * Licensed under the Apache License, Version 2.0.
 */

#include <gtest/gtest.h>
#include "share/system_variable/ob_sys_var_meta.h"

namespace oceanbase {
namespace share {
namespace {

TEST(SystemVariableRegistration, competition_variables_are_resolvable)
{
  const char *names[] = {"enable_file_sql", "ai_column_strict_read"};
  const ObSysVarClassType ids[] = {
    SYS_VAR_ENABLE_FILE_SQL,
    SYS_VAR_AI_COLUMN_STRICT_READ,
  };
  for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); ++i) {
    const ObString name = ObString::make_string(names[i]);
    int64_t store_idx = -1;
    EXPECT_EQ(ids[i], ObSysVarMeta::find_sys_var_id_by_name(name));
    EXPECT_EQ(OB_SUCCESS, ObSysVarMeta::calc_sys_var_store_idx_by_name(name, store_idx));
    EXPECT_TRUE(ObSysVarMeta::is_valid_sys_var_store_idx(store_idx));
    EXPECT_EQ(ObString::make_string(names[i]),
              ObSysVarMeta::get_sys_var_name_by_id(ids[i]));
  }
}

} // namespace
} // namespace share
} // namespace oceanbase
