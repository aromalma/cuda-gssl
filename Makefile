# Compiler
NVCC := nvcc
# Compiler flags
NVCCFLAGS := -O3 -shared -Xcompiler -fPIC $(shell python3 -m pybind11 --includes)
# NVCCFLAGS := -O3 -shared -std=c++13 -Xcompiler -fPIC $(shell python3 -m pybind11 --includes)
# NVCCFLAGS := -O3 -shared -std=c++11 -Xcompiler -fPIC $(shell python3 -m pybind11 --includes)

# Source files
CUDA_SRCS := gssl.cu kernels.cu
# Object files
CUDA_OBJS := $(CUDA_SRCS:.cu=.o)
# Executable name
EXEC := gssl$(shell python3-config --extension-suffix)

# Compute capabilities
GENCODE := -gencode=arch=compute_60,code=sm_60 \
           -gencode=arch=compute_61,code=sm_61 \
           -gencode=arch=compute_70,code=sm_70 \
           -gencode=arch=compute_75,code=sm_75 \
           -gencode=arch=compute_80,code=sm_80 \
           -gencode=arch=compute_86,code=sm_86 \
           -gencode=arch=compute_87,code=sm_87 \
           -gencode=arch=compute_86,code=compute_86

# Main target
all: $(EXEC)

# Rule to build the executable
$(EXEC): $(CUDA_OBJS)
	$(NVCC) $(GENCODE) $(NVCCFLAGS) $(CUDA_OBJS)  -o $(EXEC)

# Rule to build object files from CUDA source files
%.o: %.cu
	$(NVCC) $(GENCODE) $(NVCCFLAGS)   -c $< -o $@

# Clean command to remove object files and the executable
clean:
	rm -f $(CUDA_OBJS) $(EXEC)

# Phony target to avoid conflicts with files named "all" or "clean"
.PHONY: all clean
