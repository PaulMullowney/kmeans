#include "KmeansCudaKernels.h"

#include "loadVector.hcu"
#include "reduction.hcu"
#include "multiply.hcu"

template<class TYPE, class VTYPE, const int N_UNROLL, const int DELTA>
//__launch_bounds__(256,1)
__global__ void _dev_ClosestCentersBeginT(const TYPE * __restrict__ A, const TYPE * __restrict__ B,
					  const TYPE * __restrict__ normRowsOfA_squared,
					  const TYPE * __restrict__ normColsOfB_squared,
					  TYPE * __restrict__ C, int * __restrict__ Cindices) {

  __shared__ VTYPE Ashmem[TILESIZEY][TILESIZEX];
  __shared__ VTYPE Bshmem[TILESIZEY][TILESIZEX];

  /* read in the vector data from global memory */
  __shared__ VTYPE L2normB[TILESIZE];

  int r = blockIdx.y*N_UNROLL*TILESIZE + N_UNROLL*threadIdx.x;
  int c = blockIdx.x*N_UNROLL*TILESIZE + N_UNROLL*threadIdx.x;

  VTYPE Creg[N_UNROLL];
  for (int n=0; n<N_UNROLL; ++n)
    zero<VTYPE>(Creg[n]);

  /* matrix matrix multiply */
  TYPE * a = const_cast<TYPE *>(A) + r + threadIdx.y*dev_nRowsAPadded;
  TYPE * b = const_cast<TYPE *>(B) + c + threadIdx.y*dev_nColsB;

  multiplyT<TYPE,VTYPE,N_UNROLL,DELTA>(a, b, Creg, Ashmem, Bshmem);

  /* load the vector data */
  if (threadIdx.y==0)
    _dev_loadVector(c, dev_nColsB, normColsOfB_squared, L2normB[threadIdx.x]);
  __syncthreads();

  /* perform the partial reduction over each row in the shmem buffers */
  _dev_reduction<TILESIZEX,TILESIZEX>(c, Creg, L2normB, Ashmem, Bshmem); 

  a = reinterpret_cast<TYPE *>(&(Ashmem[threadIdx.y][0]));
  b = reinterpret_cast<TYPE *>(&(Bshmem[threadIdx.y][0]));

  /* write out the results */
  if (threadIdx.x<N_UNROLL) {
    int r = blockIdx.y*N_UNROLL*TILESIZE + N_UNROLL*threadIdx.y + threadIdx.x;
    //r += threadIdx.y;
    if(r<dev_nRowsA) {
      C[r*dev_nColsC+blockIdx.x]        = a[threadIdx.x];
      Cindices[r*dev_nColsC+blockIdx.x] = (int)b[threadIdx.x];
    }      
  }
}

template<class TYPE, class VTYPE, const int N_UNROLL, const int DELTA>
//__launch_bounds__(256,1)
__global__ void _dev_ClosestCentersBegin(const TYPE * __restrict__ A, const TYPE * __restrict__ B,
					 const TYPE * __restrict__ normRowsOfA_squared,
					 const TYPE * __restrict__ normColsOfB_squared,
					 TYPE * __restrict__ C, int * __restrict__ Cindices) {

  __shared__ VTYPE Ashmem[TILESIZEY][TILESIZEY];
  __shared__ VTYPE Bshmem[TILESIZEY][TILESIZEX];

  /* read in the vector data from global memory */
  __shared__ VTYPE L2normB[TILESIZE];

  int r = blockIdx.y*N_UNROLL*TILESIZE + N_UNROLL*threadIdx.y;
  int c = blockIdx.x*N_UNROLL*TILESIZE + N_UNROLL*threadIdx.x;

  VTYPE Creg[N_UNROLL];
  for (int n=0; n<N_UNROLL; ++n)
    zero<VTYPE>(Creg[n]);

  /* matrix matrix multiply */
  TYPE * a = const_cast<TYPE *>(A) + r*dev_nColsA + threadIdx.x;
  TYPE * b = const_cast<TYPE *>(B) + c + threadIdx.y*dev_nColsB;

  multiply<TYPE,VTYPE,N_UNROLL,DELTA>(a, b, Creg, Ashmem, Bshmem);

  /* load the vector data */
  if (threadIdx.y==0)
    _dev_loadVector(c, dev_nColsB, normColsOfB_squared, L2normB[threadIdx.x]);
  __syncthreads();

  /* perform the partial reduction over each row in the shmem buffers */
  _dev_reduction<TILESIZEY,TILESIZEX>(c, Creg, L2normB, Ashmem, Bshmem);

  a = reinterpret_cast<TYPE *>(&(Ashmem[threadIdx.y][0]));
  b = reinterpret_cast<TYPE *>(&(Bshmem[threadIdx.y][0]));

  /* write out the results */
  if (threadIdx.x<N_UNROLL) {
    r += threadIdx.x;
    if(r<dev_nRowsA) {
      C[r*dev_nColsC+blockIdx.x]        = a[threadIdx.x];
      Cindices[r*dev_nColsC+blockIdx.x] = (int)b[threadIdx.x];
    }      
  }
}



template<class TYPE, const int N_UNROLL, const int DELTA>
//__launch_bounds__(256,1)
__global__ void _dev_ClosestCentersBeginNew(const TYPE * __restrict__ A, const TYPE * __restrict__ B,
					    const TYPE * __restrict__ normRowsOfA_squared,
					    const TYPE * __restrict__ normColsOfB_squared,
					    TYPE * __restrict__ C, int * __restrict__ Cindices) {

  __shared__ TYPE Ashmem[N_UNROLL*TILESIZEY][TILESIZEY];
  __shared__ TYPE Bshmem[N_UNROLL*TILESIZEY][TILESIZEX];

  /* read in the vector data from global memory */
  __shared__ TYPE L2normB[N_UNROLL*TILESIZE];

  int r = blockIdx.y*N_UNROLL*TILESIZE + N_UNROLL*threadIdx.y;
  int c = blockIdx.x*N_UNROLL*TILESIZE + N_UNROLL*threadIdx.x;

  TYPE Creg[N_UNROLL*N_UNROLL];
  for (int n=0; n<N_UNROLL*N_UNROLL; ++n)
    Creg[n]=0;

  /* matrix matrix multiply */
  TYPE * a = const_cast<TYPE *>(A) + r*dev_nColsA + threadIdx.x;
  TYPE * b = const_cast<TYPE *>(B) + c + threadIdx.y*dev_nColsB;

  multiplyNew<TYPE,N_UNROLL,DELTA>(a, b, Creg, Ashmem, Bshmem);

  /* load the vector data */
  if (threadIdx.y==0) {
    for (int n=0; n<N_UNROLL; ++n) {
      L2normB[threadIdx.x*N_UNROLL+n] = 0;
      if (c+n<dev_nColsB)
	L2normB[threadIdx.x*N_UNROLL+n] = normColsOfB_squared[c+n];
    }
  }
  __syncthreads();

#if 0

  /* perform the partial reduction over each row in the shmem buffers */
  _dev_reduction<TYPE,TILESIZEY,TILESIZEX,N_UNROLL>(c, Creg, L2normB, Ashmem, Bshmem);

  a = reinterpret_cast<TYPE *>(&(Ashmem[threadIdx.y][0]));
  b = reinterpret_cast<TYPE *>(&(Bshmem[threadIdx.y][0]));

  /* write out the results */
  if (threadIdx.x<N_UNROLL) {
    r += threadIdx.x;
    if(r<dev_nRowsA) {
      C[r*dev_nColsC+blockIdx.x]        = a[threadIdx.x];
      Cindices[r*dev_nColsC+blockIdx.x] = (int)b[threadIdx.x];
    }      
  }
#endif
}

__host__ __device__ static __inline__ int nextPowerOfTwo(int v) {
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;
  return v;
}

template<class TYPE, const int SHMEMSIZE>
__global__ void _dev_ClosestCentersEnd(const int m0, const int n, const int k, const int SHIFT,
				       const TYPE * __restrict__ C,
				       const int * __restrict__ Cindices,
				       int * __restrict__ CindicesFinal) {

  __shared__ TYPE data[SHMEMSIZE];
  __shared__ int index[SHMEMSIZE];
  TYPE dataReg;
  TYPE indexReg;

  for (int i=blockIdx.x; i<n; i+=gridDim.x) {
    dataReg = FLT_MAX;
    indexReg = -1;

    for (int j=threadIdx.x; j<k; j+=blockDim.x) {
      TYPE val = C[i*k+j];
      int index = Cindices[i*k+j];
      if (val<dataReg) {
	dataReg = val;
	indexReg = index;
      }
    }
    data[threadIdx.x] = dataReg;
    index[threadIdx.x] = indexReg;
    __syncthreads();

    /* reduce over the block of 128 threads */
    int shift = SHIFT;
    int j;
    while (shift>=1) {
      if (threadIdx.x<shift) {
	data[threadIdx.x] = fminf(data[threadIdx.x],data[threadIdx.x+shift],
				  index[threadIdx.x],index[threadIdx.x+shift], j);
	index[threadIdx.x] = j;
      }
      __syncthreads();
      shift >>= 1;
    }
    if (threadIdx.x==0)
      CindicesFinal[m0 + i] = index[0];
    __syncthreads();
  }  
}

/* Generic Templated interface to calling the CUDA kernel */
template<class TYPE, class VTYPE,  const int N_UNROLL>
kmeansCudaErrorStatus ClosestCenters(const int m0, const int nRowsA, const int nColsA,
				     const bool isTranspose, const TYPE *A, 
				     const int nColsB, const TYPE *B, 
				     const TYPE * normRowsOfA_squared,
				     const TYPE * normColsOfB_squared,
				     const int nColsC, TYPE * C, int *Cindices,
				     int * CindicesFinal, bool& constantMemSet) {
  
  try {
    const int nBy = (nRowsA+N_UNROLL*TILESIZE-1)/(N_UNROLL*TILESIZE);
    const int nBx = (nColsB+N_UNROLL*TILESIZE-1)/(N_UNROLL*TILESIZE);
    int N = (nColsA+TILESIZE-1)/(TILESIZE);
    int delta = nColsA-(N-1)*TILESIZE;
    N-=1;
    dim3 grid = dim3(nBx, nBy);
    dim3 block = dim3(TILESIZE, TILESIZE);


    if (constantMemSet==false) {
      int nRowsAPadded = ((nRowsA + N_UNROLL*TILESIZE-1)/(N_UNROLL*TILESIZE))*N_UNROLL*TILESIZE;
      CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_N, &N, sizeof(int), 0),ERROR_CLOSESTCENTERS);
      CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_nRowsA, &nRowsA, sizeof(int)),ERROR_CLOSESTCENTERS);
      CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_nRowsAPadded, &nRowsAPadded, sizeof(int)),ERROR_CLOSESTCENTERS);
      CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_nColsA, &nColsA, sizeof(int)),ERROR_CLOSESTCENTERS);
      CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_nRowsB, &nColsA, sizeof(int)),ERROR_CLOSESTCENTERS);
      CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_nColsB, &nColsB, sizeof(int)),ERROR_CLOSESTCENTERS);
      CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_nColsC, &nColsC, sizeof(int)),ERROR_CLOSESTCENTERS);
      //int astride = TILESIZE*nColsA;
      int astride = nColsA;
      CUDA_SAFE_CALL(cudaMemcpyToSymbol(ASTRIDE, &astride, sizeof(int)),ERROR_CLOSESTCENTERS);
      constantMemSet = true;
    }
    CUDA_SAFE_CALL(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte),ERROR_CLOSESTCENTERS);
    
    if (isTranspose) {

      if (delta==1)
	_dev_ClosestCentersBeginT<TYPE,VTYPE,N_UNROLL,1><<<grid,block>>>(A,B,normRowsOfA_squared,
									 normColsOfB_squared,C,Cindices);
      else if (delta==2)
	_dev_ClosestCentersBeginT<TYPE,VTYPE,N_UNROLL,2><<<grid,block>>>(A,B,normRowsOfA_squared,
									 normColsOfB_squared,C,Cindices);
      else if (delta==3)
	_dev_ClosestCentersBeginT<TYPE,VTYPE,N_UNROLL,3><<<grid,block>>>(A,B,normRowsOfA_squared,
									 normColsOfB_squared,C,Cindices);
      else if (delta==4)
	_dev_ClosestCentersBeginT<TYPE,VTYPE,N_UNROLL,4><<<grid,block>>>(A,B,normRowsOfA_squared,
									 normColsOfB_squared,C,Cindices);
      else if (delta==5)
	_dev_ClosestCentersBeginT<TYPE,VTYPE,N_UNROLL,5><<<grid,block>>>(A,B,normRowsOfA_squared,
									 normColsOfB_squared,C,Cindices);
      else if (delta==6)
	_dev_ClosestCentersBeginT<TYPE,VTYPE,N_UNROLL,6><<<grid,block>>>(A,B,normRowsOfA_squared,
									 normColsOfB_squared,C,Cindices);
      else if (delta==7)
	_dev_ClosestCentersBeginT<TYPE,VTYPE,N_UNROLL,7><<<grid,block>>>(A,B,normRowsOfA_squared,
									 normColsOfB_squared,C,Cindices);
      else if (delta==8)
	_dev_ClosestCentersBeginT<TYPE,VTYPE,N_UNROLL,8><<<grid,block>>>(A,B,normRowsOfA_squared,
									 normColsOfB_squared,C,Cindices);
      else if (delta==9)
	_dev_ClosestCentersBeginT<TYPE,VTYPE,N_UNROLL,9><<<grid,block>>>(A,B,normRowsOfA_squared,
									 normColsOfB_squared,C,Cindices);
      else if (delta==10)
	_dev_ClosestCentersBeginT<TYPE,VTYPE,N_UNROLL,10><<<grid,block>>>(A,B,normRowsOfA_squared,
									  normColsOfB_squared,C,Cindices);
      else if (delta==11)
	_dev_ClosestCentersBeginT<TYPE,VTYPE,N_UNROLL,11><<<grid,block>>>(A,B,normRowsOfA_squared,
									  normColsOfB_squared,C,Cindices);
      else if (delta==12)
	_dev_ClosestCentersBeginT<TYPE,VTYPE,N_UNROLL,12><<<grid,block>>>(A,B,normRowsOfA_squared,
									  normColsOfB_squared,C,Cindices);
      else if (delta==13)
	_dev_ClosestCentersBeginT<TYPE,VTYPE,N_UNROLL,13><<<grid,block>>>(A,B,normRowsOfA_squared,
									  normColsOfB_squared,C,Cindices);
      else if (delta==14)
	_dev_ClosestCentersBeginT<TYPE,VTYPE,N_UNROLL,14><<<grid,block>>>(A,B,normRowsOfA_squared,
									  normColsOfB_squared,C,Cindices);
      else if (delta==15)
	_dev_ClosestCentersBeginT<TYPE,VTYPE,N_UNROLL,15><<<grid,block>>>(A,B,normRowsOfA_squared,
									  normColsOfB_squared,C,Cindices);
      else if (delta==16)
	_dev_ClosestCentersBeginT<TYPE,VTYPE,N_UNROLL,16><<<grid,block>>>(A,B,normRowsOfA_squared,
									  normColsOfB_squared,C,Cindices);
      
    } else {

      if (delta==1)
	_dev_ClosestCentersBeginNew<TYPE,N_UNROLL,1><<<grid,block>>>(A,B,normRowsOfA_squared,
									normColsOfB_squared,C,Cindices);
      else if (delta==2)
	_dev_ClosestCentersBeginNew<TYPE,N_UNROLL,2><<<grid,block>>>(A,B,normRowsOfA_squared,
									normColsOfB_squared,C,Cindices);
      else if (delta==3)
	_dev_ClosestCentersBeginNew<TYPE,N_UNROLL,3><<<grid,block>>>(A,B,normRowsOfA_squared,
									normColsOfB_squared,C,Cindices);
      else if (delta==4)
	_dev_ClosestCentersBeginNew<TYPE,N_UNROLL,4><<<grid,block>>>(A,B,normRowsOfA_squared,
									normColsOfB_squared,C,Cindices);
      else if (delta==5)
	_dev_ClosestCentersBeginNew<TYPE,N_UNROLL,5><<<grid,block>>>(A,B,normRowsOfA_squared,
									normColsOfB_squared,C,Cindices);
      else if (delta==6)
	_dev_ClosestCentersBeginNew<TYPE,N_UNROLL,6><<<grid,block>>>(A,B,normRowsOfA_squared,
									normColsOfB_squared,C,Cindices);
      else if (delta==7)
	_dev_ClosestCentersBeginNew<TYPE,N_UNROLL,7><<<grid,block>>>(A,B,normRowsOfA_squared,
									normColsOfB_squared,C,Cindices);
      else if (delta==8)
	_dev_ClosestCentersBeginNew<TYPE,N_UNROLL,8><<<grid,block>>>(A,B,normRowsOfA_squared,
									normColsOfB_squared,C,Cindices);
      else if (delta==9)
	_dev_ClosestCentersBeginNew<TYPE,N_UNROLL,9><<<grid,block>>>(A,B,normRowsOfA_squared,
									normColsOfB_squared,C,Cindices);
      else if (delta==10)
	_dev_ClosestCentersBeginNew<TYPE,N_UNROLL,10><<<grid,block>>>(A,B,normRowsOfA_squared,
									 normColsOfB_squared,C,Cindices);
      else if (delta==11)
	_dev_ClosestCentersBeginNew<TYPE,N_UNROLL,11><<<grid,block>>>(A,B,normRowsOfA_squared,
									 normColsOfB_squared,C,Cindices);
      else if (delta==12)
	_dev_ClosestCentersBeginNew<TYPE,N_UNROLL,12><<<grid,block>>>(A,B,normRowsOfA_squared,
									 normColsOfB_squared,C,Cindices);
      else if (delta==13)
	_dev_ClosestCentersBeginNew<TYPE,N_UNROLL,13><<<grid,block>>>(A,B,normRowsOfA_squared,
									 normColsOfB_squared,C,Cindices);
      else if (delta==14)
	_dev_ClosestCentersBeginNew<TYPE,N_UNROLL,14><<<grid,block>>>(A,B,normRowsOfA_squared,
									 normColsOfB_squared,C,Cindices);
      else if (delta==15)
	_dev_ClosestCentersBeginNew<TYPE,N_UNROLL,15><<<grid,block>>>(A,B,normRowsOfA_squared,
									 normColsOfB_squared,C,Cindices);
      else if (delta==16)
	_dev_ClosestCentersBeginNew<TYPE,N_UNROLL,16><<<grid,block>>>(A,B,normRowsOfA_squared,
									 normColsOfB_squared,C,Cindices);
    }

    CUDA_SAFE_CALL(cudaGetLastError(),ERROR_CLOSESTCENTERS);

    const int nThreads = 128;
    grid = dim3(getMaxConcurrentBlocks(), 1, 1);
    block = dim3(nThreads,1,1);

    int shift = nextPowerOfTwo(nColsC);
    //CUDA_SAFE_CALL(cudaFuncSetCacheConfig(_dev_ClosestCentersEnd<TYPE,nThreads>,
    //					  cudaFuncCachePreferShared),ERROR_CLOSESTCENTERS);
    _dev_ClosestCentersEnd<TYPE,nThreads><<<grid,block>>>(m0,nRowsA,nColsC,shift,C,Cindices,CindicesFinal);
    
    CUDA_SAFE_CALL(cudaGetLastError(),ERROR_CLOSESTCENTERS);
  } catch (...) {
    return ERROR_CLOSESTCENTERS;
  }
  return NO_ERROR;
}


/* Generic Templated interface to calling the CUDA kernel */
template<class TYPE>
DllExport kmeansCudaErrorStatus ClosestCenters(const int m0, const int nRowsA, const int nColsA,
					       const bool isTranspose, const TYPE *A,
					       const int nColsB, const TYPE *B, 
					       const TYPE * normRowsOfA_squared,
					       const TYPE * normColsOfB_squared,
					       const int nColsC, TYPE * C, int *Cindices,
					       int * CindicesFinal, bool& constantMemSet) {
  return NO_ERROR;
}

template<>
kmeansCudaErrorStatus ClosestCenters(const int m0, const int nRowsA, const int nColsA,
				     const bool isTranspose, const float *A,
				     const int nColsB, const float *B, 
				     const float * normRowsOfA_squared,
				     const float * normColsOfB_squared,
				     const int nColsC, float * C, int *Cindices,
				     int * CindicesFinal, bool& constantMemSet) {

  return ClosestCenters<float,FVECTOR,N_UNROLL_FLOAT>
    (m0,nRowsA,nColsA,isTranspose,(const float *)A,nColsB,(const float *)B,
     (const float *)normRowsOfA_squared, (const float *)normColsOfB_squared,
     nColsC,(float *)C,Cindices,CindicesFinal, constantMemSet);
}


template<>
kmeansCudaErrorStatus ClosestCenters(const int m0, const int nRowsA, const int nColsA,
				     const bool isTranspose, const double *A,
				     const int nColsB, const double *B, 
				     const double * normRowsOfA_squared,
				     const double * normColsOfB_squared,
				     const int nColsC, double * C, int *Cindices,
				     int * CindicesFinal, bool& constantMemSet) {
  
  return ClosestCenters<double,DVECTOR,N_UNROLL_DOUBLE>
    (m0,nRowsA,nColsA,isTranspose,(const double *)A,nColsB,(const double *)B,
     (const double *)normRowsOfA_squared, (const double *)normColsOfB_squared,
     nColsC,(double *)C,Cindices,CindicesFinal, constantMemSet);
}



/* Single precision C entry Point */
kmeansCudaErrorStatus ClosestCentersF(const int m0, const int nRowsA, const int nColsA,
				      const float *A, const int nColsB, const float *B, 
				      const float * normRowsOfA_squared,
				      const float * normColsOfB_squared,
				      const int nColsC, float * C, int *Cindices,
				      int * CindicesFinal) {
  
  bool constantMemSet = false;
  return ClosestCenters<float,FVECTOR,N_UNROLL_FLOAT>
    (m0,nRowsA,nColsA,false,A,nColsB,B,normRowsOfA_squared,
     normColsOfB_squared,nColsC,C,Cindices,CindicesFinal, constantMemSet);
}

/* Double precision C entry Point */
kmeansCudaErrorStatus ClosestCentersD(const int m0, const int nRowsA, const int nColsA, 
				      const double *A, const int nColsB, const double *B, 
				      const double * normRowsOfA_squared,
				      const double * normColsOfB_squared,
				      const int nColsC, double * C, int *Cindices,
				      int * CindicesFinal) {

  bool constantMemSet = false;
  return ClosestCenters<double,DVECTOR,N_UNROLL_DOUBLE>
    (m0,nRowsA,nColsA,false,A,nColsB,B,normRowsOfA_squared,
     normColsOfB_squared,nColsC,C,Cindices,CindicesFinal, constantMemSet);
}
