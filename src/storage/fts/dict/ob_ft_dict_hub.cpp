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

#define USING_LOG_PREFIX STORAGE_FTS

#include "storage/fts/dict/ob_ft_dict_hub.h"

#include "lib/ob_errno.h"
#include "lib/oblog/ob_log_module.h"
#include "lib/utility/ob_macro_utils.h"
#include "storage/fts/dict/ob_ft_cache_container.h"
#include "storage/fts/dict/ob_ft_dict_def.h"
#include "storage/fts/dict/ob_ft_range_dict.h"
namespace oceanbase
{
namespace storage
{
int ObFTDictHub::init()
{
  static constexpr int K_MAX_DICT_BUCKET = 128; // for now, only built-in dicts.
  int ret = OB_SUCCESS;
  if (OB_FAIL(dict_map_.create(K_MAX_DICT_BUCKET, "dict_map"))) {
  } else if (OB_FAIL(rw_dict_lock_.init(K_MAX_DICT_BUCKET))) {
  } else {
    is_inited_ = true;
  }
  return ret;
};

int ObFTDictHub::destroy()
{
  int ret = OB_SUCCESS;
  is_inited_ = false;
  for (int64_t i = 0; i < SHARED_DICT_COUNT; ++i) {
    ObFTSharedDictEntry &entry = shared_dicts_[i];
    ObFTRangeDict *dict = entry.dict_.load(std::memory_order_acquire);
    if (OB_NOT_NULL(dict)) {
      entry.dict_.store(nullptr, std::memory_order_release);
      dict->~ObFTRangeDict();
      shared_alloc_.free(dict);
    }
    if (OB_NOT_NULL(entry.container_)) {
      entry.container_->~ObFTCacheRangeContainer();
      shared_alloc_.free(entry.container_);
      entry.container_ = nullptr;
    }
  }
  shared_alloc_.reset();
  return ret;
}

int ObFTDictHub::get_shared_range_dict(const ObFTDictDesc &desc, ObFTRangeDict *&dict)
{
  int ret = OB_SUCCESS;
  dict = nullptr;
  const int64_t idx = static_cast<int64_t>(desc.type_) - 1;
  if (OB_UNLIKELY(!is_inited_)) {
    ret = OB_NOT_INIT;
  } else if (OB_UNLIKELY(idx < 0 || idx >= SHARED_DICT_COUNT)) {
    ret = OB_NOT_SUPPORTED;
  } else {
    ObFTSharedDictEntry &entry = shared_dicts_[idx];
    dict = entry.dict_.load(std::memory_order_acquire);
    if (OB_ISNULL(dict)) {
      lib::ObMutexGuard guard(entry.lock_);
      dict = entry.dict_.load(std::memory_order_acquire);
      if (OB_ISNULL(dict)) {
        ObFTCacheRangeContainer *container = nullptr;
        ObFTRangeDict *range_dict = nullptr;
        if (OB_ISNULL(container = OB_NEWx(ObFTCacheRangeContainer, &shared_alloc_, shared_alloc_))) {
          ret = OB_ALLOCATE_MEMORY_FAILED;
        } else {
          int tmp_ret = load_cache(desc, *container);
          if (OB_SUCCESS == tmp_ret) {
            // cache already registered by a previous build
          } else if (OB_ENTRY_NOT_EXIST == tmp_ret) {
            // dict info not registered yet, build from embedded dict text
            ret = build_cache(desc, *container);
          } else {
            ret = tmp_ret;
          }
        }
        if (OB_SUCC(ret)) {
          if (OB_ISNULL(range_dict = OB_NEWx(ObFTRangeDict,
                                             &shared_alloc_,
                                             shared_alloc_,
                                             container,
                                             desc))) {
            ret = OB_ALLOCATE_MEMORY_FAILED;
          } else if (OB_FAIL(range_dict->init())) {
          } else {
            entry.container_ = container;
            entry.dict_.store(range_dict, std::memory_order_release);
            dict = range_dict;
            container = nullptr;
            range_dict = nullptr;
          }
        }
        if (OB_FAIL(ret)) {
          if (OB_NOT_NULL(range_dict)) {
            range_dict->~ObFTRangeDict();
            shared_alloc_.free(range_dict);
          }
          if (OB_NOT_NULL(container)) {
            container->~ObFTCacheRangeContainer();
            shared_alloc_.free(container);
          }
        }
      }
    }
  }
  return ret;
}

int ObFTDictHub::build_cache(const ObFTDictDesc &desc, ObFTCacheRangeContainer &container)
{
  int ret = OB_SUCCESS;
  ObFTDictInfoKey key(static_cast<uint64_t>(desc.type_));
  ObFTDictInfo info;
  container.reset();

  if (!is_inited_) {
    ret = OB_NOT_INIT;
  } else {
    ObBucketHashWLockGuard guard(rw_dict_lock_, key.hash());

    // try if valid with no recursive lock
    if (OB_FAIL(get_dict_info(key, info))) {
      if (OB_HASH_NOT_EXIST == ret) {
        // dict not exist, make new one, by caller
        ret = OB_ENTRY_NOT_EXIST;
      } else {
      }
    } else if (OB_FAIL(ObFTRangeDict::try_load_cache(desc, info.range_count_, container))) {
      if (OB_ENTRY_NOT_EXIST == ret) {
      } else {
      }
    }

    if (OB_FAIL(ret)) {
      if (OB_ENTRY_NOT_EXIST == ret) {
        if (OB_FAIL(ObFTRangeDict::build_cache_from_ik_dict(desc, container))) {
        } else if (FALSE_IT(info.range_count_ = container.get_handles().size())) {
        } else if (OB_FAIL(put_dict_info(key, info))) {
        }
      }
    }
  }
  return ret;
}

int ObFTDictHub::load_cache(const ObFTDictDesc &desc, ObFTCacheRangeContainer &container)
{
  int ret = OB_SUCCESS;
  ObFTDictInfo info;
  container.reset();
  ObFTDictInfoKey key(static_cast<uint64_t>(desc.type_));
  if (!is_inited_) {
    ret = OB_NOT_INIT;
  } else {
    {
      ObBucketHashRLockGuard guard(rw_dict_lock_, key.hash());
      if (OB_FAIL(get_dict_info(key, info))) {
        if (OB_HASH_NOT_EXIST == ret) {
          // dict not exist, make new one, by caller
          ret = OB_ENTRY_NOT_EXIST;
        } else {
        }
      }
    }
    if (OB_FAIL(ret)) {
    } else if (OB_FAIL(ObFTRangeDict::try_load_cache(desc, info.range_count_, container))) {
      if (OB_ENTRY_NOT_EXIST == ret) {
        // dict not exist, make new one, by caller
      } else {
      }
    }
  }

  return ret;
}


int ObFTDictHub::get_dict_info(const ObFTDictInfoKey &key, ObFTDictInfo &info)
{
  int ret = OB_SUCCESS;

  if (OB_FAIL(dict_map_.get_refactored(key, info))) {
    if (OB_HASH_NOT_EXIST != ret) {
    }
  }

  return ret;
}

int ObFTDictHub::put_dict_info(const ObFTDictInfoKey &key, const ObFTDictInfo &info)
{
  int ret = OB_SUCCESS;
  const int cover_exist_flag = 1;
  if (OB_FAIL(dict_map_.set_refactored(key, info, cover_exist_flag))) {
  }

  return ret;
}
} //  namespace storage
} //  namespace oceanbase
