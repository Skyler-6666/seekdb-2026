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

#include <cerrno>
#include <cstring>
#include <net/if.h>
#include <net/if_arp.h>
#include <netinet/in.h>
#include <sys/ioctl.h>
#include <sys/random.h>
#include <sys/socket.h>
#include <unistd.h>

namespace oceanbase
{
namespace sql
{

// Returns a POSIX error only if neither a hardware node nor secure entropy is
// available. Keep this helper independent of SQL initialization so the failure
// paths can be tested without starting a server.
inline int init_linux_uuid_node(unsigned char (&node)[6], bool &used_random, int &mac_error)
{
  used_random = false;
  mac_error = ENODEV;
  bool found = false;
  const int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
  if (sock < 0) {
    mac_error = errno;
  } else {
    struct ifreq interfaces[32] = {};
    struct ifconf ifc = {};
    ifc.ifc_len = sizeof(interfaces);
    ifc.ifc_req = interfaces;
    if (ioctl(sock, SIOCGIFCONF, &ifc) < 0) {
      mac_error = errno;
    } else {
      const int count = ifc.ifc_len / sizeof(struct ifreq);
      for (int i = 0; i < count && !found; ++i) {
        struct ifreq interface = {};
        memcpy(interface.ifr_name, interfaces[i].ifr_name, IFNAMSIZ - 1);
        if (ioctl(sock, SIOCGIFFLAGS, &interface) < 0) {
          mac_error = errno;
        } else if (0 == (interface.ifr_flags & IFF_LOOPBACK)) {
          if (ioctl(sock, SIOCGIFHWADDR, &interface) < 0) {
            mac_error = errno;
          } else if (ARPHRD_ETHER == interface.ifr_hwaddr.sa_family) {
            memcpy(node, interface.ifr_hwaddr.sa_data, sizeof(node));
            found = true;
            mac_error = 0;
          }
        }
      }
    }
    close(sock);
  }

  int ret = 0;
  if (!found) {
    size_t offset = 0;
    while (0 == ret && offset < sizeof(node)) {
      const ssize_t count = getrandom(node + offset, sizeof(node) - offset, 0);
      if (count > 0) {
        offset += count;
      } else if (count < 0 && EINTR == errno) {
        // Signals and short reads must not leave part of the node uninitialized.
      } else {
        ret = count < 0 ? errno : EIO;
      }
    }
    if (0 == ret) {
      // RFC 9562 section 6.10: random node IDs set the multicast bit so they
      // cannot conflict with an IEEE 802 unicast hardware address.
      node[0] |= 0x01;
      used_random = true;
    }
  }
  return ret;
}

} // namespace sql
} // namespace oceanbase
