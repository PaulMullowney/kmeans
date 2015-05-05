#include "KmeansCudaKernels.h"

template<class TYPE, const int BLOCKSIZE>
__global__ void _dev_Compactness(const int m0, const int m, const int n, const int k,
				 const TYPE * __restrict__ data, 
				 const int * __restrict__ indices, 
				 const TYPE * __restrict__ centers, 
				 TYPE * __restrict__ compactness) {

  __shared__ int ind;
  __shared__ TYPE shmem[BLOCKSIZE];
  shmem[threadIdx.x] = 0;
  __syncthreads();

  for (int i=blockIdx.x; i<m; i+=gridDim.x) {
    if (threadIdx.x==0) ind = indices[m0+i];
    __syncthreads();

    for (int tidx=threadIdx.x; tidx<n; tidx+=blockDim.x) {
      //TYPE v = data[i*n+tidx] - centers[tidx*k+ind];
      TYPE v = data[i*n+tidx] - centers[ind*n + tidx];
      shmem[threadIdx.x] += v*v;
    }
    __syncthreads();
  }

  int tidx = threadIdx.x;
  int shift = BLOCKSIZE >> 1;
  while (shift>=1) {
    if (tidx<shift) shmem[tidx] += shmem[tidx+shift];
    __syncthreads();
    shift >>= 1;
  }

  if (threadIdx.x==0)
    compactness[blockIdx.x] += shmem[0];
}

template<class TYPE>
DllExport kmeansCudaErrorStatus Compactness(const int m0, const int m,
				  const int n, const int k,
				  const TYPE * data, const int * indices, 
				  const TYPE * centers, TYPE * compactness) {
  
  return NO_ERROR;
}

template<>
kmeansCudaErrorStatus Compactness(const int m0, const int m,
				  const int n, const int k,
				  const float * data, const int * indices, 
				  const float * centers, float * compactness) {
  
  try {
    const int nThreads = 128;
    dim3 grid = dim3(getMaxConcurrentBlocks(), 1, 1);
    dim3 block = dim3(nThreads,1,1);

    CUDA_SAFE_CALL(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte),ERROR_COMPACTNESS);
    CUDA_SAFE_CALL(cudaFuncSetCacheConfig(_dev_Compactness<float,nThreads>,cudaFuncCachePreferShared),ERROR_COMPACTNESS);

    _dev_Compactness<float,nThreads><<<grid,block>>>(m0,m,n,k,data,indices,centers,compactness);
    CUDA_SAFE_CALL(cudaGetLastError(),ERROR_COMPACTNESS);
    
  } catch (...) {
    return ERROR_COMPACTNESS;
  }
  return NO_ERROR;
}

template<>
kmeansCudaErrorStatus Compactness(const int m0, const int m,
				  const int n, const int k,
				  const double * data, const int * indices, 
				  const double * centers, double * compactness) {
  
  try {
    const int nThreads = 128;
    dim3 grid = dim3(getMaxConcurrentBlocks(), 1, 1);
    dim3 block = dim3(nThreads,1,1);

    CUDA_SAFE_CALL(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte),ERROR_COMPACTNESS);
    CUDA_SAFE_CALL(cudaFuncSetCacheConfig(_dev_Compactness<double,nThreads>,cudaFuncCachePreferShared),ERROR_COMPACTNESS);

    _dev_Compactness<double,nThreads><<<grid,block>>>(m0,m,n,k,data,indices,centers,compactness);
    CUDA_SAFE_CALL(cudaGetLastError(),ERROR_COMPACTNESS);
    
  } catch (...) {
    return ERROR_COMPACTNESS;
  }
  return NO_ERROR;
}

/* Single precision C entry Point */
kmeansCudaErrorStatus CompactnessF(const int m0, const int m, const int n, const int k,
				   const float * data, const int * indices, 
				   const float * centers, float * compactness) {  
  return Compactness<float>(m0,m,n,k,data,indices,centers,compactness);
}

/* Double precision C entry Point */
kmeansCudaErrorStatus CompactnessD(const int m0, const int m, const int n, const int k,
				   const double * data, const int * indices, 
				   const double * centers, double * compactness) {
  return Compactness<double>(m0,m,n,k,data,indices,centers,compactness);
}
