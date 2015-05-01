#include "KmeansCudaKernels.h"

template<class TYPE>
__global__ void _dev_ClusterCentersBegin(const int m, const int n, const int k,
					 const TYPE * __restrict__ data,
					 const int * __restrict__ indices,
					 TYPE * __restrict__ centers, 
					 int * __restrict__ counts) {

  __shared__ int ind;
  for (int i=blockIdx.x; i<m; i+=gridDim.x) {
    if (threadIdx.x==0) {
      ind = indices[i];
      counts[blockIdx.x*k + ind]++;
    }
    __syncthreads();

    for (int tidx=threadIdx.x; tidx<n; tidx+=blockDim.x) 
      centers[blockIdx.x*n*k + ind*n + tidx] += data[i*n+tidx];
    __syncthreads();
  }
}

template <class TYPE>
class SharedMem
{
public:
  // Ensure that we won't compile any un-specialized types
  __device__ TYPE* getPointer() { TYPE x; return &x; };
};
// specialization for float
template <>
class SharedMem <float>
{
public:
  __device__ float* getPointer() { extern __shared__ float rShmem[]; return rShmem; }
};
// specialization for double
template <>
class SharedMem <double>
{
public:
  __device__ double* getPointer() { extern __shared__ double dShmem[]; return dShmem; }
};

template<class TYPE>
__global__ void _dev_ClusterCentersEnd(const int m, const int n, const int k,
				       const TYPE * __restrict__ centers_large, 
				       const int * __restrict__ counts_large,
				       TYPE * __restrict__ centers, 
				       int * __restrict__ counts) {

  SharedMem<TYPE> shared;
  TYPE* shmem = shared.getPointer();

  int * count = (int *) shmem;
  TYPE * center = (TYPE*) &(shmem[1]);

  if (threadIdx.x==0) *count = 0;
  int tidx = threadIdx.x;

  while (tidx<n) {
    center[tidx]=0;
    tidx += blockDim.x;
  }
  __syncthreads();

  for (int i=0; i<m; ++i) {    
    for (tidx=threadIdx.x; tidx<n; tidx+=blockDim.x) 
      center[tidx] += centers_large[i*n*k + blockIdx.x*n + tidx];    

    if (threadIdx.x==0) *count += counts_large[i*k + blockIdx.x];
    __syncthreads();
  }

  /* write the centers to global memory */
  tidx = threadIdx.x;
  while (tidx<n) {
    //centers[tidx*k + blockIdx.x] = center[tidx]/(*count);
	centers[blockIdx.x*n + tidx] = center[tidx]/(*count);
    tidx+=blockDim.x;
  }    
  if (threadIdx.x==0){
    counts[blockIdx.x] = *count;  
  }
}

/* Generic Templated interface to calling the CUDA kernel */
template<class TYPE>
DllExport kmeansCudaErrorStatus ClusterCenters(const int m, const int n, const int k,
				     const TYPE * data, const int * indices,
				     TYPE * centers_large, int * counts_large,
				     TYPE * centers, int * counts) {
  
  return NO_ERROR;
}

/* Generic Templated interface to calling the CUDA kernel */
template<>
kmeansCudaErrorStatus ClusterCenters(const int m, const int n, const int k,
				     const float * data, const int * indices,
				     float * centers_large, int * counts_large,
				     float * centers, int * counts) {
  
  try {
    const int nThreads = 128;
    dim3 grid = dim3(getMaxConcurrentBlocks(), 1, 1);
    dim3 block = dim3(nThreads,1,1);
    const int shmemBytes = n*sizeof(float)+sizeof(int);

    CUDA_SAFE_CALL(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte),ERROR_CLUSTERCENTERS);
    CUDA_SAFE_CALL(cudaFuncSetCacheConfig(_dev_ClusterCentersBegin<float>, cudaFuncCachePreferShared),ERROR_CLUSTERCENTERS);

    _dev_ClusterCentersBegin<float><<<grid,block>>>(m,n,k,data,indices,centers_large,counts_large);
    CUDA_SAFE_CALL(cudaGetLastError(),ERROR_CLUSTERCENTERS);

    CUDA_SAFE_CALL(cudaFuncSetCacheConfig(_dev_ClusterCentersEnd<float>, cudaFuncCachePreferShared),ERROR_CLUSTERCENTERS);

    grid = dim3(k, 1, 1);
    block = dim3(nThreads,1,1);
    _dev_ClusterCentersEnd<float><<<grid,block,shmemBytes>>>(getMaxConcurrentBlocks(),n,k,centers_large,counts_large,centers,counts);    
    CUDA_SAFE_CALL(cudaGetLastError(),ERROR_CLUSTERCENTERS);
  } catch (...) {
    return ERROR_CLUSTERCENTERS;
  }
  return NO_ERROR;
}

/* Generic Templated interface to calling the CUDA kernel */
template<>
kmeansCudaErrorStatus ClusterCenters(const int m, const int n, const int k,
				     const double * data, const int * indices,
				     double * centers_large, int * counts_large,
				     double * centers, int * counts) {
  
  try {
    const int nThreads = 128;
    dim3 grid = dim3(getMaxConcurrentBlocks(), 1, 1);
    dim3 block = dim3(nThreads,1,1);
    const int shmemBytes = n*sizeof(double)+sizeof(int);

    CUDA_SAFE_CALL(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte),ERROR_CLUSTERCENTERS);
    CUDA_SAFE_CALL(cudaFuncSetCacheConfig(_dev_ClusterCentersBegin<double>, cudaFuncCachePreferShared),ERROR_CLUSTERCENTERS);

    _dev_ClusterCentersBegin<double><<<grid,block>>>(m,n,k,data,indices,centers_large,counts_large);
    CUDA_SAFE_CALL(cudaGetLastError(),ERROR_CLUSTERCENTERS);

    CUDA_SAFE_CALL(cudaFuncSetCacheConfig(_dev_ClusterCentersEnd<double>, cudaFuncCachePreferShared),ERROR_CLUSTERCENTERS);

    grid = dim3(k, 1, 1);
    block = dim3(nThreads,1,1);
    _dev_ClusterCentersEnd<double><<<grid,block,shmemBytes>>>(getMaxConcurrentBlocks(),n,k,centers_large,counts_large,centers,counts);    
    CUDA_SAFE_CALL(cudaGetLastError(),ERROR_CLUSTERCENTERS);
  } catch (...) {
    return ERROR_CLUSTERCENTERS;
  }
  return NO_ERROR;
}

/* Single precision C entry Point */
kmeansCudaErrorStatus ClusterCentersF(const int m, const int n, const int k,
				      const float * data, const int * indices,
				      float * centers_large, int * counts_large,
				      float * centers, int * counts) {
  
  return ClusterCenters<float>(m,n,k,data,indices,centers_large,
			       counts_large,centers,counts);
}

/* Double precision C entry Point */
kmeansCudaErrorStatus ClusterCentersD(const int m, const int n, const int k,
				      const double * data, const int * indices,
				      double * centers_large, int * counts_large,
				      double * centers, int * counts) {
  
  return ClusterCenters<double>(m,n,k,data,indices,centers_large,
				counts_large,centers,counts);
}
