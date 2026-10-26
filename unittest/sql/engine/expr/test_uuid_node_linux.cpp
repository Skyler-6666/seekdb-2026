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

#include <algorithm>
#include <cstdarg>
#include <gtest/gtest.h>
#include "sql/engine/expr/ob_uuid_node_linux.h"

namespace
{
struct Scenario
{
  bool fail_socket = false;
  unsigned long fail_ioctl = 0;
  bool loopback_only = false;
  bool non_ethernet = false;
  int interface_count = 1;
  bool deny_first_interface = false;
  int entropy_error = 0;
  bool empty_entropy = false;
  bool short_entropy = false;
  bool real_entropy = false;
  int close_count = 0;
  int entropy_calls = 0;
  unsigned char next_byte = 0;
};

Scenario scenario;
constexpr int TEST_SOCKET = 12345;
constexpr unsigned char HARDWARE_NODE[6] = {0x02, 0x11, 0x22, 0x33, 0x44, 0x55};

class LinuxUuidNodeTest : public ::testing::Test
{
protected:
  void SetUp() override { scenario = Scenario{}; }
  void expect_fallback()
  {
    unsigned char node[6] = {};
    bool random = false;
    int mac_error = 0;
    ASSERT_EQ(0, oceanbase::sql::init_linux_uuid_node(node, random, mac_error));
    EXPECT_TRUE(random);
    EXPECT_NE(0, mac_error);
    EXPECT_EQ(1, node[0] & 1);
    EXPECT_EQ(1, node[0]);
    for (size_t i = 1; i < sizeof(node); ++i) {
      EXPECT_EQ(i, node[i]);
    }
    EXPECT_EQ(scenario.fail_socket ? 0 : 1, scenario.close_count);
  }
};
} // namespace

extern "C" ssize_t __real_getrandom(void *, size_t, unsigned int);

extern "C" int __wrap_socket(int, int, int)
{
  if (scenario.fail_socket) {
    errno = EPERM;
    return -1;
  }
  return TEST_SOCKET;
}

extern "C" int __wrap_ioctl(int fd, unsigned long request, ...)
{
  va_list args;
  va_start(args, request);
  void *value = va_arg(args, void *);
  va_end(args);
  EXPECT_EQ(TEST_SOCKET, fd);
  if (request == scenario.fail_ioctl) {
    errno = EACCES;
    return -1;
  }
  if (SIOCGIFCONF == request) {
    auto *ifc = static_cast<struct ifconf *>(value);
    EXPECT_GE(ifc->ifc_len, static_cast<int>(scenario.interface_count * sizeof(struct ifreq)));
    ifc->ifc_len = scenario.interface_count * sizeof(struct ifreq);
    for (int i = 0; i < scenario.interface_count; ++i) {
      std::strcpy(ifc->ifc_req[i].ifr_name, i == 0 ? "eth0" : "eth1");
    }
  } else if (SIOCGIFFLAGS == request) {
    auto *ifr = static_cast<struct ifreq *>(value);
    if (scenario.deny_first_interface && 0 == std::strcmp(ifr->ifr_name, "eth0")) {
      errno = EPERM;
      return -1;
    }
    ifr->ifr_flags = scenario.loopback_only ? IFF_LOOPBACK : 0;
  } else if (SIOCGIFHWADDR == request) {
    auto *ifr = static_cast<struct ifreq *>(value);
    ifr->ifr_hwaddr.sa_family = scenario.non_ethernet ? ARPHRD_LOOPBACK : ARPHRD_ETHER;
    std::memcpy(ifr->ifr_hwaddr.sa_data, HARDWARE_NODE, sizeof(HARDWARE_NODE));
  } else {
    ADD_FAILURE() << "Unexpected ioctl request " << request;
    errno = EINVAL;
    return -1;
  }
  return 0;
}

extern "C" int __wrap_close(int fd)
{
  EXPECT_EQ(TEST_SOCKET, fd);
  ++scenario.close_count;
  return 0;
}

extern "C" ssize_t __wrap_getrandom(void *buffer, size_t length, unsigned int flags)
{
  ++scenario.entropy_calls;
  if (scenario.real_entropy) {
    return __real_getrandom(buffer, length, flags);
  } else if (scenario.entropy_error != 0) {
    errno = scenario.entropy_error;
    return -1;
  } else if (scenario.empty_entropy) {
    return 0;
  } else if (scenario.short_entropy && 1 == scenario.entropy_calls) {
    errno = EINTR;
    return -1;
  }
  const size_t count = scenario.short_entropy ? std::min(length, size_t(2)) : length;
  auto *bytes = static_cast<unsigned char *>(buffer);
  for (size_t i = 0; i < count; ++i) {
    bytes[i] = scenario.next_byte++;
  }
  return count;
}

TEST_F(LinuxUuidNodeTest, preserves_hardware_mac_without_reading_entropy)
{
  unsigned char node[6] = {};
  bool random = true;
  int mac_error = -1;
  ASSERT_EQ(0, oceanbase::sql::init_linux_uuid_node(node, random, mac_error));
  EXPECT_FALSE(random);
  EXPECT_EQ(0, mac_error);
  EXPECT_EQ(0, std::memcmp(node, HARDWARE_NODE, sizeof(node)));
  EXPECT_EQ(0, scenario.entropy_calls);
  EXPECT_EQ(1, scenario.close_count);
}

TEST_F(LinuxUuidNodeTest, socket_denied) { scenario.fail_socket = true; expect_fallback(); }
TEST_F(LinuxUuidNodeTest, enumeration_denied) { scenario.fail_ioctl = SIOCGIFCONF; expect_fallback(); }
TEST_F(LinuxUuidNodeTest, interface_flags_denied) { scenario.fail_ioctl = SIOCGIFFLAGS; expect_fallback(); }
TEST_F(LinuxUuidNodeTest, hardware_address_denied) { scenario.fail_ioctl = SIOCGIFHWADDR; expect_fallback(); }
TEST_F(LinuxUuidNodeTest, loopback_only) { scenario.loopback_only = true; expect_fallback(); }
TEST_F(LinuxUuidNodeTest, non_ethernet_address) { scenario.non_ethernet = true; expect_fallback(); }
TEST_F(LinuxUuidNodeTest, no_interfaces) { scenario.interface_count = 0; expect_fallback(); }
TEST_F(LinuxUuidNodeTest, continues_after_an_inaccessible_interface)
{
  scenario.interface_count = 2;
  scenario.deny_first_interface = true;
  unsigned char node[6] = {};
  bool random = true;
  int mac_error = 0;
  ASSERT_EQ(0, oceanbase::sql::init_linux_uuid_node(node, random, mac_error));
  EXPECT_FALSE(random);
  EXPECT_EQ(0, mac_error);
  EXPECT_EQ(0, std::memcmp(node, HARDWARE_NODE, sizeof(node)));
  EXPECT_EQ(0, scenario.entropy_calls);
  EXPECT_EQ(1, scenario.close_count);
}
TEST_F(LinuxUuidNodeTest, entropy_interrupted_and_short_reads)
{
  scenario.fail_socket = true;
  scenario.short_entropy = true;
  expect_fallback();
  EXPECT_EQ(4, scenario.entropy_calls);
}

TEST_F(LinuxUuidNodeTest, entropy_failure_is_propagated)
{
  scenario.fail_socket = true;
  scenario.entropy_error = EACCES;
  unsigned char node[6] = {};
  bool random = true;
  int mac_error = 0;
  EXPECT_EQ(EACCES, oceanbase::sql::init_linux_uuid_node(node, random, mac_error));
  EXPECT_FALSE(random);
  EXPECT_EQ(EPERM, mac_error);
}

TEST_F(LinuxUuidNodeTest, empty_entropy_read_fails)
{
  scenario.fail_socket = true;
  scenario.empty_entropy = true;
  unsigned char node[6] = {};
  bool random = true;
  int mac_error = 0;
  EXPECT_EQ(EIO, oceanbase::sql::init_linux_uuid_node(node, random, mac_error));
  EXPECT_FALSE(random);
}

TEST_F(LinuxUuidNodeTest, real_entropy_produces_distinct_multicast_nodes)
{
  scenario.fail_socket = true;
  scenario.real_entropy = true;
  unsigned char first[6] = {};
  unsigned char second[6] = {};
  bool random = false;
  int mac_error = 0;
  ASSERT_EQ(0, oceanbase::sql::init_linux_uuid_node(first, random, mac_error));
  ASSERT_TRUE(random);
  ASSERT_EQ(0, oceanbase::sql::init_linux_uuid_node(second, random, mac_error));
  EXPECT_TRUE(random);
  EXPECT_EQ(1, first[0] & 1);
  EXPECT_EQ(1, second[0] & 1);
  EXPECT_NE(0, std::memcmp(first, second, sizeof(first)));
}

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
