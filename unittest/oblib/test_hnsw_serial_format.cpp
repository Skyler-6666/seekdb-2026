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

#include <cstdint>
#include <cstring>
#include <vector>
#include <gtest/gtest.h>
#include "lib/ob_errno.h"
#include "lib/vector/ob_hnsw_serial_format.h"

using namespace oceanbase::common;
using namespace oceanbase::common::obvectorutil;

namespace
{

/* Mirrors the layout of nmslib/hnswlib v0.7.0, which is also the layout VSAG
 * reads. The test only needs a byte exact image of that layout, it does not
 * need a working graph. */
struct SerialImage
{
  static constexpr int64_t kDim = 8;
  static constexpr int64_t kCount = 5;
  static constexpr int32_t kMaxDegree = 16;

  uint64_t max_elements_;
  uint64_t cur_element_count_;
  uint64_t size_data_per_element_;
  uint64_t label_offset_;
  uint64_t offset_data_;
  uint64_t max_m_;
  uint64_t max_m0_;
  uint64_t m_;
  double mult_;
  uint64_t ef_construction_;
  std::vector<char> bytes;

  SerialImage()
      : max_elements_(kCount),
        cur_element_count_(kCount),
        size_data_per_element_(0),
        label_offset_(0),
        offset_data_(0),
        max_m_(kMaxDegree),
        max_m0_(2 * kMaxDegree),
        m_(kMaxDegree),
        mult_(1.0 / 16.0),
        ef_construction_(200)
  {
    const uint64_t size_links_level0 = max_m0_ * sizeof(int32_t) + sizeof(uint32_t);
    const uint64_t data_size = kDim * sizeof(float);
    size_data_per_element_ = size_links_level0 + data_size + sizeof(int64_t);
    offset_data_ = size_links_level0;
    label_offset_ = size_links_level0 + data_size;

    const uint64_t level0_size = cur_element_count_ * size_data_per_element_;
    bytes.assign(OB_HNSW_SERIAL_HEADER_SIZE + level0_size, 0);

    int64_t pos = 0;
    auto write_u64 = [&](uint64_t value) {
      memcpy(bytes.data() + pos, &value, sizeof(value));
      pos += sizeof(value);
    };
    auto write_i32 = [&](int32_t value) {
      memcpy(bytes.data() + pos, &value, sizeof(value));
      pos += sizeof(value);
    };
    auto write_u32 = [&](uint32_t value) {
      memcpy(bytes.data() + pos, &value, sizeof(value));
      pos += sizeof(value);
    };
    auto write_f64 = [&](double value) {
      memcpy(bytes.data() + pos, &value, sizeof(value));
      pos += sizeof(value);
    };

    write_u64(0);                          // offset_level0_
    write_u64(max_elements_);
    write_u64(cur_element_count_);
    write_u64(size_data_per_element_);
    write_u64(label_offset_);
    write_u64(offset_data_);
    write_i32(1);                          // max_level_
    write_u32(0);                          // enterpoint_node_
    write_u64(max_m_);
    write_u64(max_m0_);
    write_u64(m_);
    write_f64(mult_);
    write_u64(ef_construction_);

    ASSERT_EQ(OB_HNSW_SERIAL_HEADER_SIZE, pos);

    /* The labels of a freshly built index are the insertion order. */
    for (int64_t i = 0; i < static_cast<int64_t>(cur_element_count_); ++i) {
      int64_t label = i;
      memcpy(bytes.data() + OB_HNSW_SERIAL_HEADER_SIZE + i * size_data_per_element_ + label_offset_,
             &label,
             sizeof(label));
    }
  }

  int64_t label_at(int64_t index) const
  {
    int64_t label = -1;
    memcpy(&label,
           bytes.data() + OB_HNSW_SERIAL_HEADER_SIZE + index * size_data_per_element_ + label_offset_,
           sizeof(label));
    return label;
  }

  /* The header layout is fixed: maxM_ at 56, maxM0_ at 64 and M_ at 72. */
  void set_m_params(uint64_t max_m, uint64_t max_m0, uint64_t m)
  {
    max_m_ = max_m;
    max_m0_ = max_m0;
    m_ = m;
    memcpy(bytes.data() + 56, &max_m_, sizeof(max_m_));
    memcpy(bytes.data() + 64, &max_m0_, sizeof(max_m0_));
    memcpy(bytes.data() + 72, &m_, sizeof(m_));
  }
};

} // namespace

TEST(TestHnswSerialFormat, parse_header)
{
  SerialImage image;
  HnswSerialHeader header;
  ASSERT_EQ(OB_SUCCESS, parse_hnsw_serial_header(image.bytes.data(), image.bytes.size(), header));
  ASSERT_EQ(0, header.offset_level0_);
  ASSERT_EQ(image.max_elements_, header.max_elements_);
  ASSERT_EQ(image.cur_element_count_, header.cur_element_count_);
  ASSERT_EQ(image.size_data_per_element_, header.size_data_per_element_);
  ASSERT_EQ(image.label_offset_, header.label_offset_);
  ASSERT_EQ(image.offset_data_, header.offset_data_);
  ASSERT_EQ(1, header.max_level_);
  ASSERT_EQ(0, header.enterpoint_node_);
  ASSERT_EQ(image.max_m_, header.max_m_);
  ASSERT_EQ(image.max_m0_, header.max_m0_);
  ASSERT_EQ(image.m_, header.m_);
  ASSERT_EQ(image.mult_, header.mult_);
  ASSERT_EQ(image.ef_construction_, header.ef_construction_);
}

TEST(TestHnswSerialFormat, parse_rejects_short_buffer)
{
  SerialImage image;
  HnswSerialHeader header;
  ASSERT_NE(OB_SUCCESS, parse_hnsw_serial_header(image.bytes.data(), OB_HNSW_SERIAL_HEADER_SIZE - 1, header));
  ASSERT_NE(OB_SUCCESS, parse_hnsw_serial_header(nullptr, 1024, header));
}

TEST(TestHnswSerialFormat, check_header)
{
  SerialImage image;
  HnswSerialHeader header;
  ASSERT_EQ(OB_SUCCESS, parse_hnsw_serial_header(image.bytes.data(), image.bytes.size(), header));
  ASSERT_EQ(OB_SUCCESS, check_hnsw_serial_header(header, image.bytes.size(), SerialImage::kMaxDegree));

  /* VSAG refuses an index whose M differs from the M of the index it was
   * created with, so a mismatch has to be detected here. */
  ASSERT_NE(OB_SUCCESS, check_hnsw_serial_header(header, image.bytes.size(), SerialImage::kMaxDegree + 1));
}

TEST(TestHnswSerialFormat, check_header_rejects_broken_layout)
{
  SerialImage image;
  HnswSerialHeader header;

  image.set_m_params(32, 64, 32);
  ASSERT_EQ(OB_SUCCESS, parse_hnsw_serial_header(image.bytes.data(), image.bytes.size(), header));
  ASSERT_EQ(OB_SUCCESS, check_hnsw_serial_header(header, image.bytes.size(), 32));

  /* M_ of the serialized index has to match as well. */
  image.set_m_params(32, 64, 16);
  ASSERT_EQ(OB_SUCCESS, parse_hnsw_serial_header(image.bytes.data(), image.bytes.size(), header));
  ASSERT_NE(OB_SUCCESS, check_hnsw_serial_header(header, image.bytes.size(), 32));

  /* maxM0_ is always twice maxM_. */
  image.set_m_params(32, 33, 32);
  ASSERT_EQ(OB_SUCCESS, parse_hnsw_serial_header(image.bytes.data(), image.bytes.size(), header));
  ASSERT_NE(OB_SUCCESS, check_hnsw_serial_header(header, image.bytes.size(), 32));
}

TEST(TestHnswSerialFormat, check_header_rejects_truncated_index)
{
  SerialImage image;
  HnswSerialHeader header;
  ASSERT_EQ(OB_SUCCESS, parse_hnsw_serial_header(image.bytes.data(), image.bytes.size(), header));
  const int64_t truncated_size = OB_HNSW_SERIAL_HEADER_SIZE
                                 + static_cast<int64_t>(image.size_data_per_element_) - 1;
  ASSERT_NE(OB_SUCCESS, check_hnsw_serial_header(header, truncated_size, SerialImage::kMaxDegree));
}

TEST(TestHnswSerialFormat, patch_labels)
{
  SerialImage image;
  const int64_t ids[SerialImage::kCount] = {1001, 1002, 1003, 1004, 1005};
  ASSERT_EQ(OB_SUCCESS, patch_hnsw_serial_labels(image.bytes.data(), image.bytes.size(), ids, SerialImage::kCount));
  for (int64_t i = 0; i < SerialImage::kCount; ++i) {
    ASSERT_EQ(ids[i], image.label_at(i));
  }
}

TEST(TestHnswSerialFormat, patch_labels_rejects_wrong_count)
{
  SerialImage image;
  const int64_t ids[SerialImage::kCount] = {1001, 1002, 1003, 1004, 1005};
  ASSERT_NE(OB_SUCCESS,
            patch_hnsw_serial_labels(image.bytes.data(), image.bytes.size(), ids, SerialImage::kCount - 1));
  ASSERT_NE(OB_SUCCESS, patch_hnsw_serial_labels(image.bytes.data(), image.bytes.size(), nullptr, SerialImage::kCount));
}

/* The third party header repository only ships libgtest, the test carries its own entry point. */
int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
