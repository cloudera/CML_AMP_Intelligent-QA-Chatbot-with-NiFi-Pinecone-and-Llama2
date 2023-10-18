#!/bin/bash

# Get cuda version by torch
CUDA_VERSION=`python -c "import torch;print(torch.version.cuda.replace('.', '') if torch.cuda.is_available() else '')"`
echo $CUDA_VERSION

DEVICE="cpu"
CPU_OPT="basic"

if [ "${CUDA_VERSION}" = "" ]; then
    echo "CUDA not support, use cpu version"
else
    DEVICE="cu${CUDA_VERSION//./}"
    echo "CUDA version: $CUDA_VERSION, download path: $DEVICE"
fi

echo "Checking CPU support:"
CPU_SUPPORT=$(lscpu)

echo "$CPU_SUPPORT" | grep -q "avx "
if [ $? -eq 0 ]; then
  echo "CPU supports AVX."
  CPU_OPT="AVX"
else
  echo "CPU does not support AVX."
fi

echo "$CPU_SUPPORT" | grep -q "avx2"
if [ $? -eq 0 ]; then
  echo "CPU supports AVX2."
  CPU_OPT="AVX2"
else
  echo "CPU does not support AVX2."
fi

echo "$CPU_SUPPORT" | grep -q "avx512"
if [ $? -eq 0 ]; then
  echo "CPU supports AVX512."
  CPU_OPT="AVX512"
else
  echo "CPU does not support AVX512."
fi


# Get raw nvcc output
RAW_NVCC_OUTPUT=$(/usr/local/cuda/bin/nvcc --version)
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

python -m pip install llama-cpp-python --force-reinstall --no-cache-dir