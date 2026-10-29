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
int ObFTDictHub::init(common::ObIAllocator *allocator)
{
  static constexpr int K_MAX_DICT_BUCKET = 128; // for now, only built-in dicts.
  int ret = OB_SUCCESS;
  if (OB_ISNULL(allocator)) {
    ret = OB_INVALID_ARGUMENT;
  } else if (OB_FAIL(dict_map_.create(K_MAX_DICT_BUCKET, "dict_map"))) {
  } else if (OB_FAIL(rw_dict_lock_.init(K_MAX_DICT_BUCKET))) {
  } else if (OB_FAIL(builtin_dict_lock_.init(16))) {
  } else {
    allocator_ = allocator;
    is_inited_ = true;
  }
  return ret;
};

int ObFTDictHub::destroy()
{
  int ret = OB_SUCCESS;
  is_inited_ = false;
  BuiltinRangeDictEntry *entries[2] = {&main_entry_, &quan_entry_};
  for (BuiltinRangeDictEntry *entry : entries) {
    if (OB_NOT_NULL(entry->dict_)) {
      entry->dict_->~ObFTRangeDict();
      allocator_->free(entry->dict_);
      entry->dict_ = nullptr;
    }
    if (OB_NOT_NULL(entry->container_)) {
      entry->container_->~ObFTCacheRangeContainer();
      allocator_->free(entry->container_);
      entry->container_ = nullptr;
    }
  }
  return ret;
}

int ObFTDictHub::ensure_builtin_range_dict(const ObFTDictType type, ObFTRangeDict *&dict)
{
  int ret = OB_SUCCESS;
  dict = nullptr;
  if (!is_inited_) {
    ret = OB_NOT_INIT;
  } else if (ObFTDictType::DICT_IK_MAIN != type && ObFTDictType::DICT_IK_QUAN != type) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    BuiltinRangeDictEntry &entry = (ObFTDictType::DICT_IK_MAIN == type) ? main_entry_ : quan_entry_;
    {
      ObBucketHashRLockGuard guard(builtin_dict_lock_, static_cast<uint64_t>(type));
      dict = entry.dict_;
    }
    if (OB_NOT_NULL(dict)) {
      // fast path: already built
    } else {
      ObBucketHashWLockGuard guard(builtin_dict_lock_, static_cast<uint64_t>(type));
      if (OB_ISNULL(entry.dict_)) {
        ObFTCacheRangeContainer *container = nullptr;
        ObFTRangeDict *new_dict = nullptr;
        ObFTDictDesc desc = (ObFTDictType::DICT_IK_MAIN == type)
            ? ObFTDictDesc("main_dict",
                           ObFTDictType::DICT_IK_MAIN,
                           ObCharsetType::CHARSET_UTF8MB4,
                           ObCollationType::CS_TYPE_UTF8MB4_BIN)
            : ObFTDictDesc("quan_dict",
                           ObFTDictType::DICT_IK_QUAN,
                           ObCharsetType::CHARSET_UTF8MB4,
                           ObCollationType::CS_TYPE_UTF8MB4_BIN);
        if (OB_ISNULL(container = OB_NEWx(ObFTCacheRangeContainer, allocator_, *allocator_))) {
          ret = OB_ALLOCATE_MEMORY_FAILED;
        } else if (OB_FAIL(load_cache(desc, *container)) && OB_ENTRY_NOT_EXIST != ret) {
        } else if (OB_ENTRY_NOT_EXIST == ret && OB_FAIL(build_cache(desc, *container))) {
        } else if (OB_ISNULL(new_dict = OB_NEWx(ObFTRangeDict, allocator_, *allocator_, container, desc))) {
          ret = OB_ALLOCATE_MEMORY_FAILED;
        } else if (OB_FAIL(new_dict->init())) {
        } else {
          entry.container_ = container;
          entry.dict_ = new_dict;
          new_dict = nullptr;
          container = nullptr;
        }
        if (OB_FAIL(ret)) {
          if (OB_NOT_NULL(new_dict)) {
            new_dict->~ObFTRangeDict();
            allocator_->free(new_dict);
          }
          if (OB_NOT_NULL(container)) {
            container->~ObFTCacheRangeContainer();
            allocator_->free(container);
          }
        }
      }
      dict = entry.dict_;
    }
  }
  return ret;
}

int ObFTDictHub::get_builtin_range_dict(const ObFTDictType type, ObFTRangeDict *&dict)
{
  return ensure_builtin_range_dict(type, dict);
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
