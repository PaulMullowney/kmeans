#include "KmeansCudaKernels.h"

template<class TYPE, const int SHMEMSIZE>
__global__ void _dev_rowNormalize(const int m, const int n, 
				  const TYPE * __restrict__ input, 
				  TYPE * __restrict__ output) {

  __shared__ TYPE data[SHMEMSIZE];
  for (int i=blockIdx.x; i<m; i+=gridDim.x) {
    data[threadIdx.x] = 0.f;
    __syncthreads();
    
    for (int tid=threadIdx.x; tid<n; tid+=blockDim.x) {
      TYPE val = input[i*n + tid];
      data[threadIdx.x] += val*val;
    }
    __syncthreads();

    /* reduce over the block */
    int shift = SHMEMSIZE >> 1;
    while (shift>=1) {
      if (threadIdx.x<shift) data[threadIdx.x] += data[threadIdx.x+shift];
      __syncthreads();
      shift >>= 1;
    }

    if (threadIdx.x==0)
      output[i] = data[0];
    __syncthreads();
  }
}

template<class TYPE, const int SHMEMSIZE>
__global__ void _dev_colNormalize(const int m, const int n, 
				  const TYPE * __restrict__ input, 
				  TYPE * __restrict__ output) {
  
  int tid=threadIdx.x + blockIdx.x * blockDim.x;
  TYPE res = 0.0;
  if (tid<n) {
    for (int i = 0; i<m; ++i) {
      TYPE val = input[i*n + tid];
      res += val*val;
    }
    output[tid] = res;
  }
}


/* Generic Templated interface to calling the CUDA kernel */
template<class TYPE>
DllExport kmeansCudaErrorStatus rowNormalize(const int m, const int n, 
				   const TYPE * __restrict__ input, 
				   TYPE * __restrict__ output) {
  return NO_ERROR;
}

template<>
DllExport kmeansCudaErrorStatus rowNormalize(const int m, const int n, 
				   const float * __restrict__ input, 
				   float * __restrict__ output) {
  try {
    const int nThreads = 128;
    dim3 grid = dim3(getMaxConcurrentBlocks(), 1, 1);
    dim3 block = dim3(nThreads,1,1);
    CUDA_SAFE_CALL(cudaFuncSetCacheConfig(_dev_rowNormalize<float,nThreads>, 
					  cudaFuncCachePreferShared),ERROR_ROWNORMALIZE);
    CUDA_SAFE_CALL(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte),ERROR_ROWNORMALIZE);

    _dev_rowNormalize<float,nThreads><<<grid,block>>>(m,n,input,output);
    CUDA_SAFE_CALL(cudaGetLastError(),ERROR_ROWNORMALIZE);

  } catch (...) {
    return ERROR_ROWNORMALIZE;
  }
  return NO_ERROR;
}


template<>
DllExport kmeansCudaErrorStatus rowNormalize(const int m, const int n, 
				   const double * __restrict__ input, 
				   double * __restrict__ output) {
  try {
    const int nThreads = 128;
    dim3 grid = dim3(getMaxConcurrentBlocks(), 1, 1);
    dim3 block = dim3(nThreads,1,1);
    CUDA_SAFE_CALL(cudaFuncSetCacheConfig(_dev_rowNormalize<double,nThreads>, 
					  cudaFuncCachePreferShared),ERROR_ROWNORMALIZE);
    CUDA_SAFE_CALL(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte),ERROR_ROWNORMALIZE);

    _dev_rowNormalize<double,nThreads><<<grid,block>>>(m,n,input,output);
    CUDA_SAFE_CALL(cudaGetLastError(),ERROR_ROWNORMALIZE);

  } catch (...) {
    return ERROR_ROWNORMALIZE;
  }
  return NO_ERROR;
}


/* Single precision C entry Point */
kmeansCudaErrorStatus rowNormalizeF(const int m, const int n, 
				    const float * __restrict__ input, 
				    float * __restrict__ output) {
  return rowNormalize<float>(m,n,input,output);
}

/* Double precision C entry Point */
kmeansCudaErrorStatus rowNormalizeD(const int m, const int n, 
				    const double * __restrict__ input, 
				    double * __restrict__ output) {
  return rowNormalize<double>(m,n,input,output);
}


/* Generic Templated interface to calling the CUDA kernel */
template<class TYPE>
DllExport kmeansCudaErrorStatus colNormalize(const int m, const int n, 
					     const TYPE * __restrict__ input, 
					     TYPE * __restrict__ output) {
  return NO_ERROR;
}

template<>
DllExport kmeansCudaErrorStatus colNormalize(const int m, const int n, 
					     const float * __restrict__ input, 
					     float * __restrict__ output) {
  try {
    const int nThreads = 128;
    dim3 grid = dim3(getMaxConcurrentBlocks(), 1, 1);
    dim3 block = dim3(nThreads,1,1);
    CUDA_SAFE_CALL(cudaFuncSetCacheConfig(_dev_colNormalize<float,nThreads>, 
					  cudaFuncCachePreferShared),ERROR_COLNORMALIZE);
    CUDA_SAFE_CALL(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte),ERROR_COLNORMALIZE);

    _dev_colNormalize<float,nThreads><<<grid,block>>>(m,n,input,output);
    CUDA_SAFE_CALL(cudaGetLastError(),ERROR_COLNORMALIZE);

  } catch (...) {
    return ERROR_COLNORMALIZE;
  }
  return NO_ERROR;
}

template<>
DllExport kmeansCudaErrorStatus colNormalize(const int m, const int n, 
					     const double * __restrict__ input, 
					     double * __restrict__ output) {
  try {
    const int nThreads = 128;
    dim3 grid = dim3(getMaxConcurrentBlocks(), 1, 1);
    dim3 block = dim3(nThreads,1,1);
    CUDA_SAFE_CALL(cudaFuncSetCacheConfig(_dev_colNormalize<double,nThreads>, 
					  cudaFuncCachePreferShared),ERROR_COLNORMALIZE);
    CUDA_SAFE_CALL(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte),ERROR_COLNORMALIZE);

    _dev_colNormalize<double,nThreads><<<grid,block>>>(m,n,input,output);
    CUDA_SAFE_CALL(cudaGetLastError(),ERROR_COLNORMALIZE);

  } catch (...) {
    return ERROR_COLNORMALIZE;
  }
  return NO_ERROR;
}

/* Single precision C entry Point */
kmeansCudaErrorStatus colNormalizeF(const int m, const int n, 
				    const float * __restrict__ input, 
				    float * __restrict__ output) {
  return colNormalize<float>(m,n,input,output);
}

/* Double precision C entry Point */
kmeansCudaErrorStatus colNormalizeD(const int m, const int n, 
				    const double * __restrict__ input, 
				    double * __restrict__ output) {
  return colNormalize<double>(m,n,input,output);
}
