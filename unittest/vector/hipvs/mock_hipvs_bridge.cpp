/*
 * Copyright (c) 2026 OceanBase.
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

/**
 * @file mock_hipvs_bridge.cpp
 *
 * A mock implementation of the seekdb <-> hipVS bridge ABI.
 *
 * It exists so that the loader logic can be exercised on a machine without
 * ROCm. The mock can be compiled into different variants to reproduce the
 * failure modes the loader has to cope with:
 *
 *   - `-DMOCK_ABI_VERSION=<n>`     report a different ABI version
 *   - `-DMOCK_NO_GPU`              report that no GPU is available
 *   - `-DMOCK_OMIT=<symbol_name>`  leave one symbol out of the shared library
 *
 * The mock implements a tiny in-memory index so that the round trip
 * build -> search -> serialize -> deserialize can be checked as well.
 */

#include "lib/vector/hipvs/seekdb_hipvs_bridge.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifndef MOCK_ABI_VERSION
#define MOCK_ABI_VERSION SEEKDB_HIPVS_BRIDGE_ABI_VERSION
#endif

namespace {

const char SEEKDB_HIPVS_PAYLOAD_MAGIC[8] = {'S', 'K', 'H', 'I', 'P', 'V', 'S', '1'};
const int64_t SEEKDB_HIPVS_PAYLOAD_HEADER = 24;

struct MockIndex
{
  seekdb_hipvs_index_params_t params{};
  std::vector<float> vectors;
  std::vector<int64_t> ids;
};

const char *const MOCK_VERSION = "mock-0.0.1";

void copy_string(char *buf, int32_t buf_size, const char *value)
{
  if (buf == nullptr || buf_size <= 0) {
    return;
  }
  strncpy(buf, value, static_cast<size_t>(buf_size - 1));
  buf[buf_size - 1] = '\0';
}

} // namespace

extern "C" {

int32_t seekdb_hipvs_abi_version(void) { return MOCK_ABI_VERSION; }

void seekdb_hipvs_version(char *buf, int32_t buf_size) { copy_string(buf, buf_size, MOCK_VERSION); }

int32_t seekdb_hipvs_gpu_info(char *buf, int32_t buf_size)
{
#ifdef MOCK_NO_GPU
  copy_string(buf, buf_size, "no gpu");
  return SEEKDB_HIPVS_ERR_NO_GPU;
#else
  copy_string(buf, buf_size, "mock device 0");
  return SEEKDB_HIPVS_OK;
#endif
}

int32_t seekdb_hipvs_resources_create(void **res)
{
  if (res == nullptr) {
    return SEEKDB_HIPVS_ERR_INVALID_ARGUMENT;
  }
  *res = reinterpret_cast<void *>(static_cast<intptr_t>(0x1));
  return SEEKDB_HIPVS_OK;
}

int32_t seekdb_hipvs_resources_destroy(void *res)
{
  (void)res;
  return SEEKDB_HIPVS_OK;
}

int32_t seekdb_hipvs_index_create(const seekdb_hipvs_index_params_t *params, void **idx)
{
  if (params == nullptr || idx == nullptr) {
    return SEEKDB_HIPVS_ERR_INVALID_ARGUMENT;
  }
  MockIndex *index = new MockIndex();
  index->params = *params;
  *idx = index;
  return SEEKDB_HIPVS_OK;
}

int32_t seekdb_hipvs_index_destroy(void *idx)
{
  delete static_cast<MockIndex *>(idx);
  return SEEKDB_HIPVS_OK;
}

#ifndef MOCK_OMIT_INDEX_BUILD
int32_t seekdb_hipvs_index_build(void *idx, const float *vectors, int64_t dim, int64_t count,
                                  const int64_t *ids)
{
  if (idx == nullptr || vectors == nullptr || dim <= 0 || count <= 0) {
    return SEEKDB_HIPVS_ERR_INVALID_ARGUMENT;
  }
  MockIndex *index = static_cast<MockIndex *>(idx);
  index->vectors.assign(vectors, vectors + count * dim);
  index->ids.resize(static_cast<size_t>(count));
  for (int64_t i = 0; i < count; ++i) {
    index->ids[static_cast<size_t>(i)] = ids == nullptr ? i : ids[i];
  }
  return SEEKDB_HIPVS_OK;
}
#endif

#ifndef MOCK_OMIT_INDEX_ADD
int32_t seekdb_hipvs_index_add(void *idx, const float *vectors, int64_t dim, int64_t count,
                                const int64_t *ids)
{
  if (idx == nullptr || vectors == nullptr || dim <= 0 || count <= 0) {
    return SEEKDB_HIPVS_ERR_INVALID_ARGUMENT;
  }
  MockIndex *index = static_cast<MockIndex *>(idx);
  const size_t offset = index->ids.size();
  index->vectors.insert(index->vectors.end(), vectors, vectors + count * dim);
  for (int64_t i = 0; i < count; ++i) {
    index->ids.push_back(ids == nullptr ? static_cast<int64_t>(offset) + i : ids[i]);
  }
  return SEEKDB_HIPVS_OK;
}
#endif

int32_t seekdb_hipvs_index_search(void *idx, const float *query, int64_t dim, int64_t topk, int32_t ef,
                                  int64_t *ids, float *distances, int64_t *result_size)
{
  (void)ef;
  if (idx == nullptr || query == nullptr || dim <= 0 || topk <= 0) {
    return SEEKDB_HIPVS_ERR_INVALID_ARGUMENT;
  }
  MockIndex *index = static_cast<MockIndex *>(idx);
  const int64_t count = static_cast<int64_t>(index->ids.size());
  if (count == 0) {
    if (result_size != nullptr) {
      *result_size = 0;
    }
    return SEEKDB_HIPVS_OK;
  }
  /* The mock returns the vectors in insertion order; the test only checks
   * that the plumbing works, not the ranking. */
  const int64_t written = count < topk ? count : topk;
  for (int64_t i = 0; i < written; ++i) {
    float distance = 0.0f;
    for (int64_t j = 0; j < dim; ++j) {
      const float diff = query[j] - index->vectors[static_cast<size_t>(i * dim + j)];
      distance += diff * diff;
    }
    distances[i] = distance;
    /* The bridge answers with the external id, not with the ordinal. */
    ids[i] = index->ids[static_cast<size_t>(i)];
  }
  if (result_size != nullptr) {
    *result_size = written;
  }
  return SEEKDB_HIPVS_OK;
}

int32_t seekdb_hipvs_index_num_elements(void *idx, int64_t *num_elements)
{
  if (idx == nullptr || num_elements == nullptr) {
    return SEEKDB_HIPVS_ERR_INVALID_ARGUMENT;
  }
  *num_elements = static_cast<int64_t>(static_cast<MockIndex *>(idx)->ids.size());
  return SEEKDB_HIPVS_OK;
}

int32_t seekdb_hipvs_index_vid_bound(void *idx, int64_t *min_vid, int64_t *max_vid)
{
  if (idx == nullptr || min_vid == nullptr || max_vid == nullptr) {
    return SEEKDB_HIPVS_ERR_INVALID_ARGUMENT;
  }
  const std::vector<int64_t> &ids = static_cast<MockIndex *>(idx)->ids;
  if (ids.empty()) {
    *min_vid = 0;
    *max_vid = 0;
    return SEEKDB_HIPVS_OK;
  }
  int64_t low = ids[0];
  int64_t high = ids[0];
  for (size_t i = 1; i < ids.size(); ++i) {
    if (ids[i] < low) {
      low = ids[i];
    }
    if (ids[i] > high) {
      high = ids[i];
    }
  }
  *min_vid = low;
  *max_vid = high;
  return SEEKDB_HIPVS_OK;
}

int32_t seekdb_hipvs_index_serialize(void *idx, char **buf, int64_t *len)
{
  if (idx == nullptr || buf == nullptr || len == nullptr) {
    return SEEKDB_HIPVS_ERR_INVALID_ARGUMENT;
  }
  MockIndex *index = static_cast<MockIndex *>(idx);
  /* Layout: magic(8) | id count(8) | dim(4) | flags(4) | ids | vectors */
  const int64_t id_count = static_cast<int64_t>(index->ids.size());
  const int64_t id_bytes = id_count * static_cast<int64_t>(sizeof(int64_t));
  const int64_t float_bytes = static_cast<int64_t>(index->vectors.size() * sizeof(float));
  const int64_t payload = SEEKDB_HIPVS_PAYLOAD_HEADER + id_bytes + float_bytes;
  char *out = static_cast<char *>(malloc(static_cast<size_t>(payload)));
  if (out == nullptr) {
    return SEEKDB_HIPVS_ERR_NO_MEMORY;
  }
  memcpy(out, SEEKDB_HIPVS_PAYLOAD_MAGIC, sizeof(SEEKDB_HIPVS_PAYLOAD_MAGIC));
  memcpy(out + 8, &id_count, sizeof(id_count));
  const int32_t dim = index->params.dim;
  memcpy(out + 16, &dim, sizeof(dim));
  const int32_t flags = 0;
  memcpy(out + 20, &flags, sizeof(flags));
  if (id_bytes > 0) {
    memcpy(out + SEEKDB_HIPVS_PAYLOAD_HEADER, index->ids.data(), static_cast<size_t>(id_bytes));
  }
  if (float_bytes > 0) {
    memcpy(out + SEEKDB_HIPVS_PAYLOAD_HEADER + id_bytes, index->vectors.data(),
           static_cast<size_t>(float_bytes));
  }
  *buf = out;
  *len = payload;
  return SEEKDB_HIPVS_OK;
}

int32_t seekdb_hipvs_index_deserialize(const char *buf, int64_t len, const seekdb_hipvs_index_params_t *params,
                                        void **idx)
{
  if (buf == nullptr || len < SEEKDB_HIPVS_PAYLOAD_HEADER || params == nullptr || idx == nullptr) {
    return SEEKDB_HIPVS_ERR_INVALID_ARGUMENT;
  }
  if (memcmp(buf, SEEKDB_HIPVS_PAYLOAD_MAGIC, sizeof(SEEKDB_HIPVS_PAYLOAD_MAGIC)) != 0) {
    return SEEKDB_HIPVS_ERR_IO;
  }
  MockIndex *index = new MockIndex();
  index->params = *params;
  int64_t id_count = 0;
  int32_t dim = 0;
  int32_t flags = 0;
  memcpy(&id_count, buf + 8, sizeof(id_count));
  memcpy(&dim, buf + 16, sizeof(dim));
  memcpy(&flags, buf + 20, sizeof(flags));
  (void)flags;
  const int64_t id_bytes = id_count * static_cast<int64_t>(sizeof(int64_t));
  const int64_t float_bytes = len - SEEKDB_HIPVS_PAYLOAD_HEADER - id_bytes;
  if (id_count < 0 || float_bytes < 0 || dim <= 0) {
    delete index;
    return SEEKDB_HIPVS_ERR_IO;
  }
  index->ids.resize(static_cast<size_t>(id_count));
  if (id_bytes > 0) {
    memcpy(index->ids.data(), buf + SEEKDB_HIPVS_PAYLOAD_HEADER, static_cast<size_t>(id_bytes));
  }
  index->vectors.resize(static_cast<size_t>(float_bytes / sizeof(float)));
  if (float_bytes > 0) {
    memcpy(index->vectors.data(), buf + SEEKDB_HIPVS_PAYLOAD_HEADER + id_bytes,
           static_cast<size_t>(float_bytes));
  }
  *idx = index;
  return SEEKDB_HIPVS_OK;
}

void seekdb_hipvs_buffer_free(char *buf) { free(buf); }

} // extern "C"
