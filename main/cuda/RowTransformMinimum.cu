#include "KmeansCudaKernels.h"

template<class TYPE, const int SHMEMSIZE>
__global__ void _dev_rowTransformMinimum(const int n, const int k, 
					 const TYPE * __restrict__ normRowsOfA_squared,
					 const TYPE * __restrict__ normColsOfB_squared,
					 const TYPE * __restrict__ input, 
					 int * __restrict__ output) {

  __shared__ TYPE data[SHMEMSIZE];
  __shared__ int index[SHMEMSIZE];
  __shared__ TYPE rowData;
  TYPE dataReg;
  TYPE indexReg;
  int m = (k+blockDim.x-1)/blockDim.x;

  for (int i=blockIdx.x; i<n; i+=gridDim.x) {
    dataReg = FLT_MAX;
    indexReg = -1;

    if (threadIdx.x==0) rowData = normRowsOfA_squared[i];
    __syncthreads();

    for (int j=0; j<m; ++j) {
      int tid = threadIdx.x + j*blockDim.x;
      if (tid<k) {
        TYPE val = input[i*k+tid];
        val = rowData + normColsOfB_squared[tid] - 2.0*val;
        if (val<dataReg) {
          dataReg = val;
          indexReg = tid;
        }
      }
    }
    data[threadIdx.x] = dataReg;
    index[threadIdx.x] = indexReg;
    __syncthreads();

    /* reduce over the block of 128 threads */
    int shift = SHMEMSIZE >> 1;
    while (shift>=1) {
      if (threadIdx.x<shift) {
        if (data[threadIdx.x+shift]<data[threadIdx.x]) {
          data[threadIdx.x] = data[threadIdx.x+shift];
          index[threadIdx.x] = index[threadIdx.x+shift];
        } else if (data[threadIdx.x+shift]==data[threadIdx.x] &&
		   index[threadIdx.x+shift] < index[threadIdx.x]) {
          data[threadIdx.x] = data[threadIdx.x+shift];
          index[threadIdx.x] = index[threadIdx.x+shift];
	}
      }
      __syncthreads();
      shift >>= 1;
    }
    if (threadIdx.x==0)
      output[i] = index[0];
  }
}


/* Generic Templated interface to calling the CUDA kernel */
template<class TYPE>
DllExport kmeansCudaErrorStatus rowTransformMinimum(const int n, const int k, 
					  const TYPE * normRowsOfA_squared, 
					  const TYPE * normColsOfB_squared, 
					  const TYPE * input, int * output) {
  try {
    const int nThreads = 128;
    dim3 grid = dim3(getMaxConcurrentBlocks(), 1, 1);
    dim3 block = dim3(nThreads,1,1);
    CUDA_SAFE_CALL(cudaFuncSetCacheConfig(_dev_rowTransformMinimum<TYPE,nThreads>, 
					  cudaFuncCachePreferShared),ERROR_ROWTRANSFORMMINIMUM);
    CUDA_SAFE_CALL(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte),ERROR_ROWTRANSFORMMINIMUM);

    _dev_rowTransformMinimum<TYPE,nThreads><<<grid,block>>>
      (n,k,normRowsOfA_squared,normColsOfB_squared,input,output);
    CUDA_SAFE_CALL(cudaGetLastError(),ERROR_ROWTRANSFORMMINIMUM);

  } catch (...) {
    return ERROR_ROWTRANSFORMMINIMUM;
  }
  return NO_ERROR;
}

/* Single precision C entry Point */
kmeansCudaErrorStatus rowTransformMinimumF(const int n, const int k, 
					   const float * normRowsOfA_squared, 
					   const float * normColsOfB_squared, 
					   const float * input, int * output) {
  return rowTransformMinimum<float>(n,k,normRowsOfA_squared,normColsOfB_squared,input,output);
}

/* Double precision C entry Point */
kmeansCudaErrorStatus rowTransformMinimumD(const int n, const int k,
					   const double * normRowsOfA_squared, 
					   const double * normColsOfB_squared, 
					   const double * input, int * output) {
  return rowTransformMinimum<double>(n,k,normRowsOfA_squared,normColsOfB_squared,input,output);
}
