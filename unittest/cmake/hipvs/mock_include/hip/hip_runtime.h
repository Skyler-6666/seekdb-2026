// Minimal stand-in for the real <hip/hip_runtime.h>. The mock hipVS install
// only needs the type names, enum constants, and function declarations that
// cuvs/cuda_runtime.h maps the CUDA names onto; nothing here is ever called,
// so declarations without definitions are fine.
#ifndef HIP_HIP_RUNTIME_H_
#define HIP_HIP_RUNTIME_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  hipSuccess = 0,
  hipErrorInvalidValue = 1,
  hipErrorMemoryAllocation = 2,
  hipErrorNotReady = 34,
  hipErrorNoDevice = 100,
  hipErrorInvalidDevice = 101,
  hipErrorNotSupported = 801
} hipError_t;

typedef int hipDataType;
typedef int hipDeviceAttribute_t;
typedef int hipFuncAttribute;
typedef int hipMemAllocationHandleType;
typedef int hipMemPoolAttr;
typedef void* hipEvent_t;
typedef void* hipStream_t;
typedef void* hipMemPool_t;

typedef struct {
  char unused[128];
} hipDeviceProp_t;

typedef struct {
  char unused[64];
} hipFuncAttributes;

typedef struct {
  char unused[64];
} hipMemPoolProps;

typedef struct {
  int type;
  void* devicePointer;
  void* hostPointer;
} hipPointerAttribute_t;

typedef enum {
  hipMemcpyHostToHost = 0,
  hipMemcpyHostToDevice = 1,
  hipMemcpyDeviceToHost = 2,
  hipMemcpyDeviceToDevice = 3,
  hipMemcpyDefault = 4
} hipMemcpyKind;

enum {
  hipDeviceAttributeComputeCapabilityMajor = 0,
  hipDeviceAttributeComputeCapabilityMinor = 1,
  hipDeviceAttributeHostRegisterReadOnlySupported = 2,
  hipDeviceAttributeL2CacheSize = 3,
  hipDeviceAttributeMaxSharedMemoryPerBlock = 4,
  hipDeviceAttributeMaxThreadsPerBlock = 5,
  hipDeviceAttributeMemoryPoolsSupported = 6,
  hipDevAttrMemoryPoolSupportedHandleTypes = 7,
  hipDeviceAttributeMultiprocessorCount = 8,
  hipDeviceAttributeClockRate = 9,
  hipDeviceAttributeMemoryClockRate = 10
};

enum {
  hipEventDisableTiming = 0x02,
  hipFuncAttributeMaxDynamicSharedMemorySize = 0,
  hipFuncAttributePreferredSharedMemoryCarveout = 1,
  hipFuncCachePreferL1 = 0x01,
  hipFuncCachePreferShared = 0x03
};

enum {
  hipHostRegisterDefault = 0x00,
  hipHostRegisterMapped = 0x02,
  hipHostRegisterIoMemory = 0x04,
  hipHostRegisterReadOnly = 0x08
};

enum {
  hipMemAllocationTypePinned = 0x04,
  hipMemHandleTypeNone = 0x00,
  hipMemLocationTypeDevice = 0x01,
  hipMemoryTypeUnregistered = 0x00,
  hipMemoryTypeHost = 0x01,
  hipMemoryTypeDevice = 0x02,
  hipMemoryTypeManaged = 0x03,
  hipMemPoolAttrReleaseThreshold = 0,
  hipMemPoolReuseAllowOpportunistic = 1,
  hipStreamNonBlocking = 0x01,
  hipStreamDefault = 0x00
};

#define hipStreamPerThread ((hipStream_t)0x02)
#define hipMemset(a, b, c) (void)0

hipError_t hipDeviceGetAttribute(int* pi, hipDeviceAttribute_t attr, int device);
hipError_t hipDeviceGetDefaultMemPool(hipMemPool_t* mem_pool, int device);
hipError_t hipDeviceSynchronize(void);
hipError_t hipDriverGetVersion(int* driverVersion);
hipError_t hipRuntimeGetVersion(int* runtimeVersion);
hipError_t hipEventCreate(hipEvent_t* event);
hipError_t hipEventCreateWithFlags(hipEvent_t* event, unsigned flags);
hipError_t hipEventDestroy(hipEvent_t event);
hipError_t hipEventElapsedTime(float* ms, hipEvent_t start, hipEvent_t stop);
hipError_t hipEventQuery(hipEvent_t event);
hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream);
hipError_t hipEventSynchronize(hipEvent_t event);
hipError_t hipFree(void* ptr);
hipError_t hipFreeAsync(void* ptr, hipStream_t stream);
hipError_t hipHostFree(void* ptr);
hipError_t hipFuncGetAttributes(hipFuncAttributes* attr, const void* func);
hipError_t hipFuncSetCacheConfig(const void* func, int config);
hipError_t hipGetDevice(int* deviceId);
hipError_t hipGetDeviceCount(int* count);
hipError_t hipGetDeviceProperties(hipDeviceProp_t* prop, int device);
const char* hipGetErrorName(hipError_t hip_error);
const char* hipGetErrorString(hipError_t hip_error);
hipError_t hipGetLastError(void);
hipError_t hipHostGetDevicePointer(void** devPtr, void* hstPtr, unsigned flags);
hipError_t hipHostUnregister(void* hostPtr);
hipError_t hipLaunchHostFunc(hipStream_t stream, void (*fn)(void*), void* userData);
hipError_t hipLaunchKernel(const void* func, uint32_t gx, uint32_t gy, uint32_t gz,
                           uint32_t bx, uint32_t by, uint32_t bz, uint32_t sharedMem,
                           hipStream_t stream, void** kernelParams, void** extra);
hipError_t hipMalloc(void** ptr, size_t size);
hipError_t hipMallocAsync(void** ptr, size_t size, hipStream_t stream);
hipError_t hipMallocFromPoolAsync(void** ptr, size_t size, hipMemPool_t mem_pool,
                                  hipStream_t stream);
hipError_t hipHostMalloc(void** ptr, size_t size, unsigned int flags);
hipError_t hipMallocManaged(void** ptr, size_t size, unsigned int flags);
hipError_t hipMemcpy(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind);
hipError_t hipMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch,
                            size_t width, size_t height, hipMemcpyKind kind,
                            hipStream_t stream);
hipError_t hipMemcpyAsync(void* dst, const void* src, size_t sizeBytes,
                          hipMemcpyKind kind, hipStream_t stream);
hipError_t hipMemGetInfo(size_t* free, size_t* total);
hipError_t hipMemPoolCreate(hipMemPool_t* mem_pool, int* pool_props);
hipError_t hipMemPoolDestroy(hipMemPool_t mem_pool);
hipError_t hipMemPoolSetAttribute(hipMemPool_t mem_pool, hipMemPoolAttr attr, void* value);
hipError_t hipMemsetAsync(void* dst, int value, size_t sizeBytes, hipStream_t stream);
hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, const void* f,
                                                        int blockSize, size_t dynamicSMemSize);
hipError_t hipOccupancyMaxPotentialBlockSize(int* minGridSize, int* blockSize,
                                             const void* f, size_t dynamicSMemSize,
                                             int blockSizeLimit);
hipError_t hipOccupancyMaxPotentialBlockSizeVariableSMem(int* minGridSize, int* blockSize,
                                                         const void* f,
                                                         size_t (*sfunc)(int),
                                                         int blockSizeLimit);
hipError_t hipPeekAtLastError(void);
hipError_t hipPointerGetAttributes(hipPointerAttribute_t* attributes, const void* ptr);
hipError_t hipSetDevice(int deviceId);
hipError_t hipStreamCreate(hipStream_t* stream);
hipError_t hipStreamCreateWithFlags(hipStream_t* stream, unsigned int flags);
hipError_t hipStreamDestroy(hipStream_t stream);
hipError_t hipStreamSynchronize(hipStream_t stream);
hipError_t hipStreamWaitEvent(hipStream_t stream, hipEvent_t event, unsigned int flags);
hipError_t hipLaunchCooperativeKernel(const void* f, uint32_t gx, uint32_t gy, uint32_t gz,
                                      uint32_t bx, uint32_t by, uint32_t bz,
                                      uint32_t sharedMem, hipStream_t stream,
                                      void** kernelParams);

#ifdef __cplusplus
}
#endif

#endif
