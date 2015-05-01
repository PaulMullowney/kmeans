#include "KmeansCudaKernels.h"

template<class TYPE, const int BLOCKSIZE>
__global__ void _dev_Compactness(const int m, const int n, const int k,
				 const TYPE * __restrict__ data, 
				 const int * __restrict__ indices, 
				 const TYPE * __restrict__ centers, 
				 TYPE * __restrict__ compactness) {

  __shared__ int ind;
  __shared__ TYPE shmem[BLOCKSIZE];
  shmem[threadIdx.x] = 0;
  __syncthreads();

  for (int i=blockIdx.x; i<m; i+=gridDim.x) {
    if (threadIdx.x==0) ind = indices[i];
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
DllExport kmeansCudaErrorStatus Compactness(const int m, const int n, const int k,
				  const TYPE * data, const int * indices, 
				  const TYPE * centers, TYPE * compactness,
				  TYPE * compactness_cpu) {
  
  try {
    const int nThreads = 128;
    dim3 grid = dim3(getMaxConcurrentBlocks(), 1, 1);
    dim3 block = dim3(nThreads,1,1);

    CUDA_SAFE_CALL(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte),ERROR_COMPACTNESS);
    CUDA_SAFE_CALL(cudaFuncSetCacheConfig(_dev_Compactness<TYPE,nThreads>,cudaFuncCachePreferShared),ERROR_COMPACTNESS);

    _dev_Compactness<TYPE,nThreads><<<grid,block>>>(m,n,k,data,indices,centers,compactness);
    CUDA_SAFE_CALL(cudaGetLastError(),ERROR_COMPACTNESS);

    /* copy back to the host and finish the calculation */
    CUDA_SAFE_CALL(cudaMemcpy(compactness_cpu,compactness,getMaxConcurrentBlocks()*sizeof(TYPE),
			      cudaMemcpyDeviceToHost),ERROR_COMPACTNESS);
    for (int j=1; j<getMaxConcurrentBlocks(); ++j)
      compactness_cpu[0] += compactness_cpu[j];
    
  } catch (...) {
    return ERROR_COMPACTNESS;
  }
  return NO_ERROR;
}

/* Single precision C entry Point */
kmeansCudaErrorStatus CompactnessF(const int m, const int n, const int k,
				   const float * data, const int * indices, 
				   const float * centers, float * compactness,
				   float * compactness_cpu) {  
  return Compactness<float>(m,n,k,data,indices,centers,compactness,compactness_cpu);
}

/* Double precision C entry Point */
kmeansCudaErrorStatus CompactnessD(const int m, const int n, const int k,
				   const double * data, const int * indices, 
				   const double * centers, double * compactness,
				   double * compactness_cpu) {
  return Compactness<double>(m,n,k,data,indices,centers,compactness,compactness_cpu);
}
