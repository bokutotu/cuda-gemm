NVCC ?= nvcc
CXXFLAGS ?= -O3 -std=c++17
# CXXFLAGS ?= -std=c++17
LDFLAGS ?=

# Link cuBLAS
# Try to add CUDA lib path if discoverable
CUDA_HOME ?= $(shell dirname $$(dirname $$(which $(NVCC))) 2>/dev/null)
ifneq ($(strip $(CUDA_HOME)),)
  LDFLAGS += -L$(CUDA_HOME)/lib64
endif
LDFLAGS += -lcublas

# --- Auto-detect GPU arch (SM) and add matching codegen flags ---
# You can override detection by passing SM on the command line, e.g.:
#   make SM=89            # build for sm_89 explicitly
#   make SM="86 89"       # fatbin for sm_86 and sm_89
# If detection fails, we build without -arch/-gencode and rely on driver JIT.

# Query compute capability via nvidia-smi (e.g., "8.9" -> "89").
DETECTED_SMS := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
                  | tr -d ' \r' | tr -d '.' | sed '/^$$/d' | sort -u)

# Allow manual override
ifneq ($(strip $(SM)),)
  GPU_SMS := $(SM)
else
  GPU_SMS := $(DETECTED_SMS)
endif

# Compose -gencode flags for each detected SM and keep PTX for forward-compat.
ifeq ($(strip $(GPU_SMS)),)
  NV_ARCH_FLAGS :=
  $(info [Makefile] No GPU SM detected; building without -arch/-gencode.)
else
  NV_ARCH_FLAGS := $(foreach sm,$(GPU_SMS),-gencode arch=compute_$(sm),code=sm_$(sm) -gencode arch=compute_$(sm),code=compute_$(sm))
  $(info [Makefile] Target GPU SM(s): $(GPU_SMS))
endif

SRCS := main.cu
OBJS := $(SRCS:.cu=.o)
DEPS := $(OBJS:.o=.d)
TARGET := gemm_test

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(CXXFLAGS) $(NV_ARCH_FLAGS) -o $@ $^ $(LDFLAGS)

# Compile .cu to .o and emit a matching .d depfile that tracks .h/.cuh
%.o: %.cu
	$(NVCC) $(CXXFLAGS) $(NV_ARCH_FLAGS) -MMD -MP -c $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) $(OBJS) $(DEPS)

-include $(DEPS)

.PHONY: all run clean
