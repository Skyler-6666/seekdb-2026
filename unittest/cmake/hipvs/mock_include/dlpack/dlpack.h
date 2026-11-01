#ifndef DLPACK_DLPACK_H_
#define DLPACK_DLPACK_H_
#include <stddef.h>
#include <stdint.h>
typedef enum { kDLCPU = 1 } DLDeviceType;
typedef struct {
  DLDeviceType device_type;
  int device_id;
} DLDevice;
typedef struct {
  uint8_t code;
  uint8_t bits;
  uint16_t lanes;
} DLDataType;
typedef struct {
  void *data;
  DLDevice device;
  int32_t ndim;
  DLDataType dtype;
  int64_t *shape;
  int64_t *strides;
  uint64_t byte_offset;
} DLTensor;
typedef struct {
  DLTensor dl_tensor;
  void *manager_ctx;
  void (*deleter)(void *);
} DLManagedTensor;
enum { kDLInt = 0U, kDLUInt = 1U, kDLFloat = 2U };
#endif
