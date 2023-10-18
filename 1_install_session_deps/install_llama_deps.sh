#!/bin/bash

# Get raw nvcc output
RAW_NVCC_OUTPUT=$(nvcc --version)
echo "Raw NVCC Output: $RAW_NVCC_OUTPUT"

# Filter the line that contains the release version
FILTERED_NVCC_OUTPUT=$(echo "$RAW_NVCC_OUTPUT" | grep "release")
echo "Filtered NVCC Output: $FILTERED_NVCC_OUTPUT"

# Extract the full CUDA version
FULL_CUDA_VERSION=$(echo "$FILTERED_NVCC_OUTPUT" | awk '{print $6}' | cut -c2-)
echo "Full CUDA Version: $FULL_CUDA_VERSION"

# Remove the second period and anything after it
CUDA_VERSION=$(echo "$FULL_CUDA_VERSION" | awk -F'.' '{print $1 "." $2}')
echo "Processed CUDA Version: $CUDA_VERSION"


if [ -z "$CUDA_VERSION" ]; then
  echo "Failed to extract CUDA version."
else
  # Construct CMAKE_CUDA_COMPILER path
  CMAKE_CUDA_COMPILER="/usr/local/cuda-${CUDA_VERSION}/bin/nvcc"

  # Export the path
  export CMAKE_CUDA_COMPILER

  # Print it out for verification
  echo "Setting CMAKE_CUDA_COMPILER to $CMAKE_CUDA_COMPILER"
fi

export CUDACXX=$CMAKE_CUDA_COMPILER
export CMAKE_ARGS="-DLLAMA_CUBLAS=on -DCMAKE_CUDA_ARCHITECTURES=native"
export FORCE_CMAKE=1

## Build llama-cpp-python w/ CUDA enablement
pip install llama-cpp-python==0.2.11 --force-reinstall --no-cache-dir