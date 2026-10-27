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

#include <stdint.h>

namespace oceanbase
{
namespace common
{
namespace obvectorutil
{

/**
 * Fixed header of a serialized HNSW index.
 *
 * The nmslib/hnswlib v0.7.0 layout and the layout used by VSAG's HNSW index are
 * byte identical for this header and for the level-0 element block, which is why
 * a graph produced by hipVS can be handed over to VSAG as-is. The only field
 * that has to be rewritten is the external label of every element: hnswlib
 * labels elements with their insertion order (0..N-1) while seekdb has to
 * address them by the row vid.
 */
struct HnswSerialHeader
{
  uint64_t offset_level0_;
  uint64_t max_elements_;
  uint64_t cur_element_count_;
  uint64_t size_data_per_element_;
  uint64_t label_offset_;
  uint64_t offset_data_;
  int32_t max_level_;
  uint32_t enterpoint_node_;
  uint64_t max_m_;
  uint64_t max_m0_;
  uint64_t m_;
  double mult_;
  uint64_t ef_construction_;
};

/* Size of the fixed header above, in bytes. */
static constexpr int64_t OB_HNSW_SERIAL_HEADER_SIZE = 96;

/* Size of the external label of one element, in bytes (hnswlib labeltype / VSAG LabelType). */
static constexpr int64_t OB_HNSW_SERIAL_LABEL_SIZE = 8;

/**
 * @brief Parse the fixed header of a serialized HNSW index.
 * @param data      buffer holding the serialized index
 * @param size      number of readable bytes in data
 * @param[out] header parsed header
 */
int parse_hnsw_serial_header(const char *data, int64_t size, HnswSerialHeader &header);

/**
 * @brief Check that a parsed header describes an index that VSAG can adopt.
 *
 * VSAG rejects an index whose M differs from the M it was created with, so the
 * caller has to build the GPU graph with a graph degree of 2 * max_degree.
 *
 * @param header            parsed header
 * @param size              total size of the serialized index
 * @param expect_max_degree the max_degree the target VSAG index was created with
 */
int check_hnsw_serial_header(const HnswSerialHeader &header, int64_t size, int32_t expect_max_degree);

/**
 * @brief Overwrite the external label of every element of a serialized HNSW index.
 * @param data     serialized index, modified in place
 * @param size     number of readable bytes in data
 * @param ids      element labels to write, one per element
 * @param id_count number of elements
 */
int patch_hnsw_serial_labels(char *data, int64_t size, const int64_t *ids, int64_t id_count);

} // namespace obvectorutil
} // namespace common
} // namespace oceanbase
