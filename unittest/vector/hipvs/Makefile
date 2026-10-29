# Standalone regression test for the optional seekdb <-> AMD hipVS bridge loader.
#
# The test does not need ROCm, a GPU or a built seekdb: it compiles the loader
# and a set of mock bridges, then drives the loader through every failure mode
# it has to survive.
#
#   make -C unittest/vector/hipvs test
#
# The mock bridges are:
#
#   good    a healthy bridge
#   badabi  a bridge reporting an incompatible ABI version
#   nosym   a bridge missing seekdb_hipvs_index_build
#   nogpu   a bridge that reports that no GPU is available

CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O1 -g -Wall -Wextra
ROOT     := $(abspath ../../..)
VECTOR_DIR := $(ROOT)/src/oblib/lib/vector
OBLIB_DIR := $(ROOT)/src/oblib
INCLUDE_DIR := $(VECTOR_DIR)/hipvs
BUILD_DIR := $(CURDIR)/build

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  SUFFIX := .dylib
  SHARED_FLAGS := -dynamiclib -undefined dynamic_lookup
else
  SUFFIX := .so
  SHARED_FLAGS := -shared -fPIC
endif

# MinGW only exports the symbols that are explicitly marked, so ask the linker
# to export all of them for the mock bridges.
ifneq (,$(findstring mingw,$(shell $(CXX) -dumpmachine 2>/dev/null)))
  SHARED_FLAGS += -Wl,--export-all-symbols
endif

MOCK_PREFIX := libmock_hipvs_bridge
MOCK_CPP := mock_hipvs_bridge.cpp
TEST_CPP := seekdb_hipvs_loader_test.cpp
LOADER_CPP := $(VECTOR_DIR)/hipvs/seekdb_hipvs_loader.cpp

MOCKS := good badabi nosym nogpu

.PHONY: all test clean

all: test

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

# A healthy bridge.
$(BUILD_DIR)/$(MOCK_PREFIX)good$(SUFFIX): $(MOCK_CPP) $(INCLUDE_DIR)/seekdb_hipvs_bridge.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(SHARED_FLAGS) -I$(OBLIB_DIR) -I$(INCLUDE_DIR) -o $@ $(MOCK_CPP)

# A bridge that reports an ABI version the loader does not know.
$(BUILD_DIR)/$(MOCK_PREFIX)badabi$(SUFFIX): $(MOCK_CPP) $(INCLUDE_DIR)/seekdb_hipvs_bridge.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(SHARED_FLAGS) -DMOCK_ABI_VERSION=999 -I$(OBLIB_DIR) -I$(INCLUDE_DIR) -o $@ $(MOCK_CPP)

# A bridge that does not export seekdb_hipvs_index_build.
$(BUILD_DIR)/$(MOCK_PREFIX)nosym$(SUFFIX): $(MOCK_CPP) $(INCLUDE_DIR)/seekdb_hipvs_bridge.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(SHARED_FLAGS) -DMOCK_OMIT_INDEX_BUILD -I$(OBLIB_DIR) -I$(INCLUDE_DIR) -o $@ $(MOCK_CPP)

# A bridge that works but reports that no GPU is available.
$(BUILD_DIR)/$(MOCK_PREFIX)nogpu$(SUFFIX): $(MOCK_CPP) $(INCLUDE_DIR)/seekdb_hipvs_bridge.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(SHARED_FLAGS) -DMOCK_NO_GPU -I$(OBLIB_DIR) -I$(INCLUDE_DIR) -o $@ $(MOCK_CPP)

$(BUILD_DIR)/seekdb_hipvs_loader_test: $(TEST_CPP) $(LOADER_CPP) $(INCLUDE_DIR)/seekdb_hipvs_loader.h \
                                     $(INCLUDE_DIR)/seekdb_hipvs_bridge.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I$(OBLIB_DIR) -I$(INCLUDE_DIR) -o $@ $(TEST_CPP) $(LOADER_CPP) $(DL_LIB)

ifeq ($(findstring Linux,$(UNAME_S)),Linux)
  DL_LIB := -ldl
else
  DL_LIB :=
endif

test: $(BUILD_DIR)/seekdb_hipvs_loader_test $(addprefix $(BUILD_DIR)/$(MOCK_PREFIX),$(addsuffix $(SUFFIX),$(MOCKS)))
	SEEKDB_HIPVS_TEST_MOCK_DIR=$(BUILD_DIR) \
	SEEKDB_HIPVS_TEST_MOCK_PREFIX=$(MOCK_PREFIX) \
	SEEKDB_HIPVS_TEST_MOCK_SUFFIX=$(SUFFIX) \
	$(BUILD_DIR)/seekdb_hipvs_loader_test

clean:
	rm -rf $(BUILD_DIR)
