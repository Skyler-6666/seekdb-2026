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
 * @file seekdb_hipvs_loader.cpp
 *
 * Implementation of the optional AMD hipVS bridge loader.
 *
 * This file has no OceanBase dependency on purpose: it only depends on the
 * platform dynamic loader, so that it can be compiled and unit tested
 * anywhere.
 */

#include "lib/vector/hipvs/seekdb_hipvs_loader.h"

#include <cstdlib>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace oceanbase {
namespace common {
namespace obhipvs {

namespace {

/**
 * The bridge is an optional runtime dependency. It is searched for in the
 * places an operator would look for it.
 */
const char *const SEEKDB_HIPVS_BRIDGE_ENV = "SEEKDB_HIPVS_BRIDGE_PATH";
const char *const SEEKDB_HOME_ENV = "SEEKDB_HOME";

std::string get_env(const char *name)
{
  const char *value = getenv(name);
  return value == nullptr ? std::string() : std::string(value);
}

std::string join_path(const std::string &directory, const char *name)
{
  if (directory.empty()) {
    return std::string(name);
  }
  if (directory[directory.size() - 1] == '/') {
    return directory + name;
  }
  return directory + "/" + name;
}

#ifdef _WIN32
using LibraryHandle = HMODULE;

LibraryHandle open_library(const char *path)
{
  return LoadLibraryA(path);
}

void *find_symbol(LibraryHandle handle, const char *name)
{
  return reinterpret_cast<void *>(GetProcAddress(handle, name));
}

void close_library(LibraryHandle handle) { FreeLibrary(handle); }
#else
using LibraryHandle = void *;

LibraryHandle open_library(const char *path) { return dlopen(path, RTLD_NOW | RTLD_LOCAL); }

void *find_symbol(LibraryHandle handle, const char *name) { return dlsym(handle, name); }

void close_library(LibraryHandle handle) { dlclose(handle); }
#endif

} // namespace

const char *HipvsLoader::get_bridge_file_name()
{
#ifdef _WIN32
  return "seekdb_hipvs_bridge.dll";
#elif defined(__APPLE__)
  return "libseekdb_hipvs_bridge.dylib";
#else
  return "libseekdb_hipvs_bridge.so";
#endif
}

HipvsLoader::~HipvsLoader() { unload(); }

HipvsLoader &HipvsLoader::instance()
{
  static HipvsLoader loader;
  static bool initialized = false;
  if (!initialized) {
    initialized = true;
    const char *bridge_name = get_bridge_file_name();

    /* 1. An explicit path always wins. */
    std::string path = get_env(SEEKDB_HIPVS_BRIDGE_ENV);
    if (!path.empty() && loader.load(path)) {
      return loader;
    }

    /* 2. Next to the other seekdb runtime libraries. */
    const std::string home = get_env(SEEKDB_HOME_ENV);
    if (!home.empty()) {
      path = join_path(join_path(home, "lib"), bridge_name);
      if (loader.load(path)) {
        return loader;
      }
    }

    /* 3. Let the platform loader find it. */
    loader.load(bridge_name);
  }
  return loader;
}

bool HipvsLoader::load(const std::string &path)
{
  unload();
  if (path.empty()) {
    error_ = "empty bridge path";
    return false;
  }
  LibraryHandle handle = open_library(path.c_str());
  if (handle == nullptr) {
    error_ = "cannot open hipvs bridge: " + path;
    return false;
  }
  handle_ = handle;

  if (!resolve_symbols()) {
    /* `resolve_symbols` already reported which symbol is missing; keep that
     * detail, it is the only actionable information for the operator. */
    if (error_.empty()) {
      error_ = "hipvs bridge " + path + " does not implement the expected ABI";
    } else {
      error_ += " (" + path + ")";
    }
    unload();
    return false;
  }

  const int32_t abi = bridge_.abi_version();
  if (abi != SEEKDB_HIPVS_BRIDGE_ABI_VERSION) {
    error_ = "hipvs bridge ABI mismatch, expected " + std::to_string(SEEKDB_HIPVS_BRIDGE_ABI_VERSION) +
             " but got " + std::to_string(abi);
    unload();
    return false;
  }

  loaded_ = true;

  char buffer[256];
  memset(buffer, 0, sizeof(buffer));
  bridge_.version(buffer, static_cast<int32_t>(sizeof(buffer)));
  version_ = buffer;

  memset(buffer, 0, sizeof(buffer));
  const int32_t gpu_status = bridge_.gpu_info(buffer, static_cast<int32_t>(sizeof(buffer)));
  gpu_info_ = buffer;
  has_gpu_ = (gpu_status == SEEKDB_HIPVS_OK);
  if (!has_gpu_) {
    error_ = "no usable AMD GPU found: " + gpu_info_;
  }
  return true;
}

bool HipvsLoader::resolve_symbols()
{
  struct Symbol
  {
    const char *name;
    void **target;
  };

  const Symbol symbols[] = {
      {"seekdb_hipvs_abi_version", reinterpret_cast<void **>(&bridge_.abi_version)},
      {"seekdb_hipvs_version", reinterpret_cast<void **>(&bridge_.version)},
      {"seekdb_hipvs_gpu_info", reinterpret_cast<void **>(&bridge_.gpu_info)},
      {"seekdb_hipvs_resources_create", reinterpret_cast<void **>(&bridge_.resources_create)},
      {"seekdb_hipvs_resources_destroy", reinterpret_cast<void **>(&bridge_.resources_destroy)},
      {"seekdb_hipvs_index_create", reinterpret_cast<void **>(&bridge_.index_create)},
      {"seekdb_hipvs_index_destroy", reinterpret_cast<void **>(&bridge_.index_destroy)},
      {"seekdb_hipvs_index_build", reinterpret_cast<void **>(&bridge_.index_build)},
      {"seekdb_hipvs_index_add", reinterpret_cast<void **>(&bridge_.index_add)},
      {"seekdb_hipvs_index_search", reinterpret_cast<void **>(&bridge_.index_search)},
      {"seekdb_hipvs_index_num_elements", reinterpret_cast<void **>(&bridge_.index_num_elements)},
      {"seekdb_hipvs_index_vid_bound", reinterpret_cast<void **>(&bridge_.index_vid_bound)},
      {"seekdb_hipvs_index_serialize", reinterpret_cast<void **>(&bridge_.index_serialize)},
      {"seekdb_hipvs_index_deserialize", reinterpret_cast<void **>(&bridge_.index_deserialize)},
      {"seekdb_hipvs_buffer_free", reinterpret_cast<void **>(&bridge_.buffer_free)},
  };

  for (size_t i = 0; i < sizeof(symbols) / sizeof(symbols[0]); ++i) {
    void *symbol = find_symbol(static_cast<LibraryHandle>(handle_), symbols[i].name);
    if (symbol == nullptr) {
      error_ = std::string("missing hipvs bridge symbol: ") + symbols[i].name;
      return false;
    }
    *symbols[i].target = symbol;
  }
  return true;
}

void HipvsLoader::unload()
{
  if (handle_ != nullptr) {
    close_library(static_cast<LibraryHandle>(handle_));
    handle_ = nullptr;
  }
  loaded_ = false;
  has_gpu_ = false;
  version_.clear();
  gpu_info_.clear();
  bridge_ = HipvsBridge();
}

} // namespace obhipvs
} // namespace common
} // namespace oceanbase
