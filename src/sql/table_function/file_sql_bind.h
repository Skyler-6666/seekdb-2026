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

#pragma once

#include "sql/table_function/table_function.h"

namespace oceanbase
{
namespace sql
{

// Built-in file SQL table functions. IDs live in the system-defined range and
// are registered by TableFunctionRegistry at observer startup.
constexpr uint64_t READ_CSV_FUNCTION_ID = 20;
constexpr uint64_t READ_JSON_FUNCTION_ID = 21;
constexpr uint64_t READ_NDJSON_FUNCTION_ID = 22;
constexpr uint64_t READ_PARQUET_FUNCTION_ID = 23;
constexpr uint64_t GLOB_FUNCTION_ID = 24;

int bind_read_csv(ObDMLResolver &resolver, const TableFunctionCall &call, TableItem *&item);
int bind_read_json(ObDMLResolver &resolver, const TableFunctionCall &call, TableItem *&item);
int bind_read_ndjson(ObDMLResolver &resolver, const TableFunctionCall &call, TableItem *&item);
int bind_read_parquet(ObDMLResolver &resolver, const TableFunctionCall &call, TableItem *&item);
int bind_glob(ObDMLResolver &resolver, const TableFunctionCall &call, TableItem *&item);

} // namespace sql
} // namespace oceanbase
