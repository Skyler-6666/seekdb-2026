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

#define USING_LOG_PREFIX LIB

#include "lib/vector/ob_hnsw_serial_format.h"
#include <cstring>
#include "lib/oblog/ob_log.h"

namespace oceanbase
{
namespace common
{
namespace obvectorutil
{

static inline uint64_t read_u64(const char *p)
{
  uint64_t value = 0;
  memcpy(&value, p, sizeof(value));
  return value;
}

static inline int32_t read_i32(const char *p)
{
  int32_t value = 0;
  memcpy(&value, p, sizeof(value));
  return value;
}

static inline uint32_t read_u32(const char *p)
{
  uint32_t value = 0;
  memcpy(&value, p, sizeof(value));
  return value;
}

static inline double read_f64(const char *p)
{
  double value = 0.0;
  memcpy(&value, p, sizeof(value));
  return value;
}

int parse_hnsw_serial_header(const char *data, int64_t size, HnswSerialHeader &header)
{
  int ret = OB_SUCCESS;
  if (data == nullptr || size < OB_HNSW_SERIAL_HEADER_SIZE) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    const char *p = data;
    header.offset_level0_ = read_u64(p);
    p += sizeof(uint64_t);
    header.max_elements_ = read_u64(p);
    p += sizeof(uint64_t);
    header.cur_element_count_ = read_u64(p);
    p += sizeof(uint64_t);
    header.size_data_per_element_ = read_u64(p);
    p += sizeof(uint64_t);
    header.label_offset_ = read_u64(p);
    p += sizeof(uint64_t);
    header.offset_data_ = read_u64(p);
    p += sizeof(uint64_t);
    header.max_level_ = read_i32(p);
    p += sizeof(int32_t);
    header.enterpoint_node_ = read_u32(p);
    p += sizeof(uint32_t);
    header.max_m_ = read_u64(p);
    p += sizeof(uint64_t);
    header.max_m0_ = read_u64(p);
    p += sizeof(uint64_t);
    header.m_ = read_u64(p);
    p += sizeof(uint64_t);
    header.mult_ = read_f64(p);
    p += sizeof(double);
    header.ef_construction_ = read_u64(p);
  }
  return ret;
}

int check_hnsw_serial_header(const HnswSerialHeader &header, int64_t size, int32_t expect_max_degree)
{
  int ret = OB_SUCCESS;
  if (header.cur_element_count_ > header.max_elements_) {
    ret = OB_INVALID_ARGUMENT;
    LOG_WARN("[OBGPUHNSW] invalid element count in serialized index", K(header.cur_element_count_),
             K(header.max_elements_));
  } else if (header.size_data_per_element_ == 0
             || header.label_offset_ > header.size_data_per_element_
             || header.size_data_per_element_ - header.label_offset_ < OB_HNSW_SERIAL_LABEL_SIZE) {
    ret = OB_INVALID_ARGUMENT;
    LOG_WARN("[OBGPUHNSW] invalid element layout in serialized index", K(header.size_data_per_element_),
             K(header.label_offset_));
  } else if (header.max_m_ != static_cast<uint64_t>(expect_max_degree)
             || header.m_ != static_cast<uint64_t>(expect_max_degree)) {
    // VSAG refuses to load an index whose M differs from the M of the index it
    // was created with, so the mismatch has to be detected before handing the
    // bytes over.
    ret = OB_INVALID_ARGUMENT;
    LOG_WARN("[OBGPUHNSW] max_degree mismatch between gpu index and vsag index", K(header.max_m_), K(header.m_),
             K(expect_max_degree));
  } else if (header.max_m0_ != header.max_m_ * 2) {
    ret = OB_INVALID_ARGUMENT;
    LOG_WARN("[OBGPUHNSW] unexpected max_m0 in serialized index", K(header.max_m0_), K(header.max_m_));
  } else {
    const int64_t level0_size = static_cast<int64_t>(header.cur_element_count_)
                                 * static_cast<int64_t>(header.size_data_per_element_);
    if (OB_HNSW_SERIAL_HEADER_SIZE + level0_size > size) {
      ret = OB_INVALID_ARGUMENT;
      LOG_WARN("[OBGPUHNSW] truncated serialized index", K(size), K(level0_size), K(header.cur_element_count_));
    }
  }
  return ret;
}

int patch_hnsw_serial_labels(char *data, int64_t size, const int64_t *ids, int64_t id_count)
{
  int ret = OB_SUCCESS;
  HnswSerialHeader header;
  if (OB_FAIL(parse_hnsw_serial_header(data, size, header))) {
  } else if (ids == nullptr || id_count != static_cast<int64_t>(header.cur_element_count_)) {
    ret = OB_INVALID_ARGUMENT;
    LOG_WARN("[OBGPUHNSW] label count does not match element count", K(id_count), K(header.cur_element_count_));
  } else if (header.size_data_per_element_ == 0
             || header.label_offset_ > header.size_data_per_element_
             || header.size_data_per_element_ - header.label_offset_ < OB_HNSW_SERIAL_LABEL_SIZE) {
    ret = OB_INVALID_ARGUMENT;
    LOG_WARN("[OBGPUHNSW] invalid element layout in serialized index", K(header.size_data_per_element_),
             K(header.label_offset_));
  } else {
    const int64_t element_size = static_cast<int64_t>(header.size_data_per_element_);
    const int64_t label_offset = static_cast<int64_t>(header.label_offset_);
    char *base = data + OB_HNSW_SERIAL_HEADER_SIZE;
    for (int64_t i = 0; OB_SUCC(ret) && i < id_count; ++i) {
      if (OB_HNSW_SERIAL_HEADER_SIZE + (i + 1) * element_size > size) {
        ret = OB_INVALID_ARGUMENT;
        LOG_WARN("[OBGPUHNSW] truncated serialized index while patching labels", K(size), K(i), K(element_size));
        break;
      }
      memcpy(base + i * element_size + label_offset, &ids[i], sizeof(int64_t));
    }
  }
  return ret;
}

} // namespace obvectorutil
} // namespace common
} // namespace oceanbase
