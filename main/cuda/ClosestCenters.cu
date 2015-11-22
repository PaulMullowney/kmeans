#include "KmeansCudaKernels.h"

#include "loadVector.hcu"
#include "reduction.hcu"
#include "multiply.hcu"

template<class TYPE, class VTYPE, const int DELTA, const int N_UNROLL>
__launch_bounds__(256,N_BLOCKS)
__global__ void _dev_ClosestCentersStripedBegin(const TYPE * __restrict__ A, const TYPE * __restrict__ B,
						const TYPE * __restrict__ normColsOfB_squared,
						TYPE * __restrict__ C, int * __restrict__ Cindices) {
  __shared__ VTYPE Ashmem[TILESIZE][TILESIZEY];
  __shared__ VTYPE Bshmem[TILESIZE][TILESIZEX];

  /* load the vector data */
  if (threadIdx.y<N_UNROLL) {
    TYPE  * z = reinterpret_cast<TYPE *>(&(Ashmem[0][0]));
    int tid = threadIdx.x + threadIdx.y*blockDim.x;
    int t = blockIdx.x*N_UNROLL*TILESIZE+tid;
    if (t<dev_nColsB) z[tid] = -0.5f*normColsOfB_squared[t];
    else z[tid] = -FLT_MAX;
  }
  __syncthreads();
 
  VTYPE Creg[N_UNROLL*(N_UNROLL<6 ? 2 : 1)];
  for (int n=0; n<N_UNROLL; ++n) {
    Creg[n] = Ashmem[0][threadIdx.x];
    if (N_UNROLL<6) Creg[n+N_UNROLL] = Creg[n];
  }
  __syncthreads();

  int r = blockIdx.y*N_UNROLL*TILESIZE + N_UNROLL*threadIdx.y;
  int c = blockIdx.x*N_UNROLL*TILESIZE + N_UNROLL*threadIdx.x;
  
  /* matrix matrix multiply */
  TYPE * a = const_cast<TYPE *>(A) + r*TILESIZE + threadIdx.x;
  TYPE * b = const_cast<TYPE *>(B) + c + threadIdx.y*dev_nColsBPadded;

  multiplyStriped<TYPE,VTYPE,N_UNROLL,DELTA>(a, b, Creg, Ashmem, Bshmem);

  /* perform the partial reduction over each row in the shmem buffers */
  _dev_reduction<TILESIZEY,TILESIZEX>(r, c, Creg, Ashmem, Bshmem, C, Cindices);
}


template<class TYPE, class VTYPE, const int DELTA, const int N_UNROLL>
__launch_bounds__(256,N_BLOCKS)
__global__ void _dev_ClosestCentersBegin(const TYPE * __restrict__ A, const TYPE * __restrict__ B,
					 const TYPE * __restrict__ normColsOfB_squared,
					 TYPE * __restrict__ C, int * __restrict__ Cindices) {

  __shared__ VTYPE Ashmem[TILESIZE][TILESIZEY];
  __shared__ VTYPE Bshmem[TILESIZE][TILESIZEX];

  /* load the vector data */
  if (threadIdx.y<N_UNROLL) {
    TYPE  * z = reinterpret_cast<TYPE *>(&(Ashmem[0][0]));
    int tid = threadIdx.x + threadIdx.y*blockDim.x;
    int t = blockIdx.x*N_UNROLL*TILESIZE+tid;
    if (t<dev_nColsB) z[tid] = -0.5f*normColsOfB_squared[t];
    else z[tid] = -FLT_MAX;
  }
  __syncthreads();

  VTYPE Creg[N_UNROLL*(N_UNROLL<6 ? 2 : 1)];
  for (int n=0; n<N_UNROLL; ++n) {
    Creg[n] = Ashmem[0][threadIdx.x];
    if (N_UNROLL<6) Creg[n+N_UNROLL] = Creg[n];
  }
  __syncthreads();

  int r = blockIdx.y*N_UNROLL*TILESIZE + N_UNROLL*threadIdx.y;
  int c = blockIdx.x*N_UNROLL*TILESIZE + N_UNROLL*threadIdx.x;
  
  /* matrix matrix multiply */
  TYPE * a = const_cast<TYPE *>(A) + r*dev_nColsA + threadIdx.x;
  TYPE * b = const_cast<TYPE *>(B) + c + threadIdx.y*dev_nColsBPadded;

  multiply<TYPE,VTYPE,N_UNROLL,DELTA>(a, b, Creg, Ashmem, Bshmem);

  /* perform the partial reduction over each row in the shmem buffers */
  _dev_reduction<TILESIZEY,TILESIZEX>(r, c, Creg, Ashmem, Bshmem, C, Cindices);
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

template<class TYPE>
inline __device__ TYPE fmin(TYPE a, TYPE b, int ia, int ib, int& i)
{
  if (a<=b) {i=ia; return a; }
  else {i=ib; return b; }
}

template<class TYPE, const int SHMEMSIZEX, const int SHMEMSIZEY>
__global__ void _dev_ClosestCentersEnd(const int m0, const int n, const int k, const int SHIFT,
				       const TYPE * __restrict__ C,
				       const int * __restrict__ Cindices,
				       int * __restrict__ CindicesFinal) {

  __shared__ TYPE data[SHMEMSIZEY][SHMEMSIZEX];
  __shared__ int index[SHMEMSIZEY][SHMEMSIZEX];
  TYPE dataReg;
  TYPE indexReg;

  for (int i=blockIdx.x*blockDim.y+threadIdx.y; i<n; i+=gridDim.x*blockDim.y) {
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
    data[threadIdx.y][threadIdx.x] = dataReg;
    index[threadIdx.y][threadIdx.x] = indexReg;
    __syncthreads();

    /* reduce over the block of 128 threads */
    int shift = SHIFT/2;
    int j;
    while (shift>=1) {
      if (threadIdx.x<shift) {
	data[threadIdx.y][threadIdx.x] = fmin<TYPE>(data[threadIdx.y][threadIdx.x],
						    data[threadIdx.y][threadIdx.x+shift],
						    index[threadIdx.y][threadIdx.x],
						    index[threadIdx.y][threadIdx.x+shift], j);
	index[threadIdx.y][threadIdx.x] = j;
      }
      __syncthreads();
      shift >>= 1;
    }
    if (threadIdx.x==0)
      CindicesFinal[m0 + i] = index[threadIdx.y][0];
    __syncthreads();
  }  
}

template<class TYPE>
__global__ void _dev_copyToAligned(const int k, const int k_padded,
				   const TYPE * in, TYPE * out) {

  for (int tid = threadIdx.x; tid<k; tid+=blockDim.x)
    out[blockIdx.x*k_padded + tid] = in[blockIdx.x*k + tid];
}

/* Generic Templated interface to calling the CUDA kernel */
template<class TYPE, class VTYPE,  const int N_UNROLL>
kmeansCudaErrorStatus ClosestCenters(const int m0, const int nRowsA, const int nColsA,
				     const bool isStriped, const TYPE *A, 
				     const int nColsB, const TYPE *B, TYPE *Bpadded, 
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
      printf("nRowsAPadded = %d\n",nRowsAPadded);
      CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_N, &N, sizeof(int), 0),ERROR_CLOSESTCENTERS);
      CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_nRowsA, &nRowsA, sizeof(int)),ERROR_CLOSESTCENTERS);
      CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_nRowsAPadded, &nRowsAPadded, sizeof(int)),ERROR_CLOSESTCENTERS);
      CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_nColsA, &nColsA, sizeof(int)),ERROR_CLOSESTCENTERS);
      CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_nRowsB, &nColsA, sizeof(int)),ERROR_CLOSESTCENTERS);
      CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_nColsB, &nColsB, sizeof(int)),ERROR_CLOSESTCENTERS);
      CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_nColsC, &nColsC, sizeof(int)),ERROR_CLOSESTCENTERS);
      int astride = nColsA;
      CUDA_SAFE_CALL(cudaMemcpyToSymbol(ASTRIDE, &astride, sizeof(int)),ERROR_CLOSESTCENTERS);
      constantMemSet = true;
    }
    CUDA_SAFE_CALL(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte),ERROR_CLOSESTCENTERS);
    CUDA_SAFE_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared),ERROR_CLOSESTCENTERS);


    const TYPE * devB = B;
    int bstride = nColsB*TILESIZE;
    if (Bpadded) {
      devB = Bpadded;
      int k_padded = 0;
      if (sizeof(TYPE) == 4)
	k_padded = ((nColsB + TILESIZE*N_UNROLL_FLOAT - 1)/(TILESIZE*N_UNROLL_FLOAT)) * TILESIZE*N_UNROLL_FLOAT;
      else
	k_padded = ((nColsB + TILESIZE*N_UNROLL_DOUBLE - 1)/(TILESIZE*N_UNROLL_DOUBLE)) * TILESIZE*N_UNROLL_DOUBLE;
      
      CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_nColsBPadded, &k_padded, sizeof(int)),ERROR_CLOSESTCENTERS);
      bstride = k_padded*TILESIZE;
      CUDA_SAFE_CALL(cudaMemcpyToSymbol(BSTRIDE, &bstride, sizeof(int)),ERROR_CLOSESTCENTERS);
      
      _dev_copyToAligned<TYPE><<<nColsA,128>>>(nColsB, k_padded, B, Bpadded);
    } else {
      CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_nColsBPadded, &nColsB, sizeof(int)),ERROR_CLOSESTCENTERS);
      CUDA_SAFE_CALL(cudaMemcpyToSymbol(BSTRIDE, &bstride, sizeof(int)),ERROR_CLOSESTCENTERS);
    }

    if (isStriped) {
      int astride = TILESIZE;
      CUDA_SAFE_CALL(cudaMemcpyToSymbol(ASTRIDE, &astride, sizeof(int)),ERROR_CLOSESTCENTERS);
      
      if (delta==1)
	_dev_ClosestCentersStripedBegin<TYPE,VTYPE,1,N_UNROLL><<<grid,block>>>(A,devB,normColsOfB_squared,C,Cindices);
      else if (delta==2)
	_dev_ClosestCentersStripedBegin<TYPE,VTYPE,2,N_UNROLL><<<grid,block>>>(A,devB,normColsOfB_squared,C,Cindices);
      else if (delta==3)
	_dev_ClosestCentersStripedBegin<TYPE,VTYPE,3,N_UNROLL><<<grid,block>>>(A,devB,normColsOfB_squared,C,Cindices);
      else if (delta==4)
	_dev_ClosestCentersStripedBegin<TYPE,VTYPE,4,N_UNROLL><<<grid,block>>>(A,devB,normColsOfB_squared,C,Cindices);
      else if (delta==5)
	_dev_ClosestCentersStripedBegin<TYPE,VTYPE,5,N_UNROLL><<<grid,block>>>(A,devB,normColsOfB_squared,C,Cindices);
      else if (delta==6)
	_dev_ClosestCentersStripedBegin<TYPE,VTYPE,6,N_UNROLL><<<grid,block>>>(A,devB,normColsOfB_squared,C,Cindices);
      else if (delta==7)
	_dev_ClosestCentersStripedBegin<TYPE,VTYPE,7,N_UNROLL><<<grid,block>>>(A,devB,normColsOfB_squared,C,Cindices);
      else if (delta==8)
	_dev_ClosestCentersStripedBegin<TYPE,VTYPE,8,N_UNROLL><<<grid,block>>>(A,devB,normColsOfB_squared,C,Cindices);
      else if (delta==9)
	_dev_ClosestCentersStripedBegin<TYPE,VTYPE,9,N_UNROLL><<<grid,block>>>(A,devB,normColsOfB_squared,C,Cindices);
      else if (delta==10)
	_dev_ClosestCentersStripedBegin<TYPE,VTYPE,10,N_UNROLL><<<grid,block>>>(A,devB,normColsOfB_squared,C,Cindices);
      else if (delta==11)
	_dev_ClosestCentersStripedBegin<TYPE,VTYPE,11,N_UNROLL><<<grid,block>>>(A,devB,normColsOfB_squared,C,Cindices);
      else if (delta==12)
	_dev_ClosestCentersStripedBegin<TYPE,VTYPE,12,N_UNROLL><<<grid,block>>>(A,devB,normColsOfB_squared,C,Cindices);
      else if (delta==13)
	_dev_ClosestCentersStripedBegin<TYPE,VTYPE,13,N_UNROLL><<<grid,block>>>(A,devB,normColsOfB_squared,C,Cindices);
      else if (delta==14)
	_dev_ClosestCentersStripedBegin<TYPE,VTYPE,14,N_UNROLL><<<grid,block>>>(A,devB,normColsOfB_squared,C,Cindices);
      else if (delta==15)
	_dev_ClosestCentersStripedBegin<TYPE,VTYPE,15,N_UNROLL><<<grid,block>>>(A,devB,normColsOfB_squared,C,Cindices);
      else if (delta==16)
	_dev_ClosestCentersStripedBegin<TYPE,VTYPE,16,N_UNROLL><<<grid,block>>>(A,devB,normColsOfB_squared,C,Cindices);

    } else {

      if (delta==1)
	_dev_ClosestCentersBegin<TYPE,VTYPE,1,N_UNROLL><<<grid,block>>>(A,devB,normColsOfB_squared,C,Cindices);
      else if (delta==2)
	_dev_ClosestCentersBegin<TYPE,VTYPE,2,N_UNROLL><<<grid,block>>>(A,devB,normColsOfB_squared,C,Cindices);
      else if (delta==3)
	_dev_ClosestCentersBegin<TYPE,VTYPE,3,N_UNROLL><<<grid,block>>>(A,devB,normColsOfB_squared,C,Cindices);
      else if (delta==4)
	_dev_ClosestCentersBegin<TYPE,VTYPE,4,N_UNROLL><<<grid,block>>>(A,devB,normColsOfB_squared,C,Cindices);
      else if (delta==5)
	_dev_ClosestCentersBegin<TYPE,VTYPE,5,N_UNROLL><<<grid,block>>>(A,devB,normColsOfB_squared,C,Cindices);
      else if (delta==6)
	_dev_ClosestCentersBegin<TYPE,VTYPE,6,N_UNROLL><<<grid,block>>>(A,devB,normColsOfB_squared,C,Cindices);
      else if (delta==7)
	_dev_ClosestCentersBegin<TYPE,VTYPE,7,N_UNROLL><<<grid,block>>>(A,devB,normColsOfB_squared,C,Cindices);
      else if (delta==8)
	_dev_ClosestCentersBegin<TYPE,VTYPE,8,N_UNROLL><<<grid,block>>>(A,devB,normColsOfB_squared,C,Cindices);
      else if (delta==9)
	_dev_ClosestCentersBegin<TYPE,VTYPE,9,N_UNROLL><<<grid,block>>>(A,devB,normColsOfB_squared,C,Cindices);
      else if (delta==10)
	_dev_ClosestCentersBegin<TYPE,VTYPE,10,N_UNROLL><<<grid,block>>>(A,devB,normColsOfB_squared,C,Cindices);
      else if (delta==11)
	_dev_ClosestCentersBegin<TYPE,VTYPE,11,N_UNROLL><<<grid,block>>>(A,devB,normColsOfB_squared,C,Cindices);
      else if (delta==12)
	_dev_ClosestCentersBegin<TYPE,VTYPE,12,N_UNROLL><<<grid,block>>>(A,devB,normColsOfB_squared,C,Cindices);
      else if (delta==13)
	_dev_ClosestCentersBegin<TYPE,VTYPE,13,N_UNROLL><<<grid,block>>>(A,devB,normColsOfB_squared,C,Cindices);
      else if (delta==14)
	_dev_ClosestCentersBegin<TYPE,VTYPE,14,N_UNROLL><<<grid,block>>>(A,devB,normColsOfB_squared,C,Cindices);
      else if (delta==15)
	_dev_ClosestCentersBegin<TYPE,VTYPE,15,N_UNROLL><<<grid,block>>>(A,devB,normColsOfB_squared,C,Cindices);
      else if (delta==16)
	_dev_ClosestCentersBegin<TYPE,VTYPE,16,N_UNROLL><<<grid,block>>>(A,devB,normColsOfB_squared,C,Cindices);
    }

    CUDA_SAFE_CALL(cudaGetLastError(),ERROR_CLOSESTCENTERS);

    const int nThreads = 128;
    grid = dim3(getMaxConcurrentBlocks(), 1, 1);
    int shift = nextPowerOfTwo(nColsC);
    block = dim3(shift,nThreads/shift,1);

    //CUDA_SAFE_CALL(cudaFuncSetCacheConfig(_dev_ClosestCentersEnd<TYPE,nThreads>,
    //					  cudaFuncCachePreferShared),ERROR_CLOSESTCENTERS);
    if (shift==2)
      _dev_ClosestCentersEnd<TYPE,2,64><<<grid,block>>>(m0,nRowsA,nColsC,shift,C,Cindices,CindicesFinal);
    else if (shift==4)
      _dev_ClosestCentersEnd<TYPE,4,32><<<grid,block>>>(m0,nRowsA,nColsC,shift,C,Cindices,CindicesFinal);
    else if (shift==8)
      _dev_ClosestCentersEnd<TYPE,8,16><<<grid,block>>>(m0,nRowsA,nColsC,shift,C,Cindices,CindicesFinal);
    else if (shift==16)
      _dev_ClosestCentersEnd<TYPE,16,8><<<grid,block>>>(m0,nRowsA,nColsC,shift,C,Cindices,CindicesFinal);
    else if (shift==32)
      _dev_ClosestCentersEnd<TYPE,32,4><<<grid,block>>>(m0,nRowsA,nColsC,shift,C,Cindices,CindicesFinal);
    else if (shift==64)
      _dev_ClosestCentersEnd<TYPE,64,2><<<grid,block>>>(m0,nRowsA,nColsC,shift,C,Cindices,CindicesFinal);
    else 
      _dev_ClosestCentersEnd<TYPE,128,1><<<grid,block>>>(m0,nRowsA,nColsC,shift,C,Cindices,CindicesFinal);
    
    CUDA_SAFE_CALL(cudaGetLastError(),ERROR_CLOSESTCENTERS);
  } catch (...) {
    return ERROR_CLOSESTCENTERS;
  }
  return NO_ERROR;
}


/* Generic Templated interface to calling the CUDA kernel */
template<class TYPE>
DllExport kmeansCudaErrorStatus ClosestCenters(const int m0, const int nRowsA, const int nColsA,
					       const bool isStriped, const TYPE *A,
					       const int nColsB, const TYPE *B, TYPE *Bpadded, 
					       const TYPE * normRowsOfA_squared,
					       const TYPE * normColsOfB_squared,
					       const int nColsC, TYPE * C, int *Cindices,
					       int * CindicesFinal, bool& constantMemSet) {
  return NO_ERROR;
}

template<>
kmeansCudaErrorStatus ClosestCenters(const int m0, const int nRowsA, const int nColsA,
				     const bool isStriped, const float *A,
				     const int nColsB, const float *B, float *Bpadded, 
				     const float * normRowsOfA_squared,
				     const float * normColsOfB_squared,
				     const int nColsC, float * C, int *Cindices,
				     int * CindicesFinal, bool& constantMemSet) {

  return ClosestCenters<float,FVECTOR,N_UNROLL_FLOAT>
    (m0,nRowsA,nColsA,isStriped,(const float *)A,nColsB,(const float *)B,(float *)Bpadded,
     (const float *)normRowsOfA_squared, (const float *)normColsOfB_squared,
     nColsC,(float *)C,Cindices,CindicesFinal, constantMemSet);
}


template<>
kmeansCudaErrorStatus ClosestCenters(const int m0, const int nRowsA, const int nColsA,
				     const bool isStriped, const double *A,
				     const int nColsB, const double *B, double *Bpadded, 
				     const double * normRowsOfA_squared,
				     const double * normColsOfB_squared,
				     const int nColsC, double * C, int *Cindices,
				     int * CindicesFinal, bool& constantMemSet) {
  
  return ClosestCenters<double,DVECTOR,N_UNROLL_DOUBLE>
    (m0,nRowsA,nColsA,isStriped,(const double *)A,nColsB,(const double *)B,(double *)Bpadded,
     (const double *)normRowsOfA_squared, (const double *)normColsOfB_squared,
     nColsC,(double *)C,Cindices,CindicesFinal, constantMemSet);
}


/* Single precision C entry Point */
kmeansCudaErrorStatus ClosestCentersF(const int m0, const int nRowsA, const int nColsA,
				      const float *A, const int nColsB, const float *B, 
				      float * Bpadded, const float * normRowsOfA_squared,
				      const float * normColsOfB_squared,
				      const int nColsC, float * C, int *Cindices,
				      int * CindicesFinal) {
  
  bool constantMemSet = false;
  return ClosestCenters<float,FVECTOR,N_UNROLL_FLOAT>
    (m0,nRowsA,nColsA,false,A,nColsB,B,Bpadded,normRowsOfA_squared,
     normColsOfB_squared,nColsC,C,Cindices,CindicesFinal, constantMemSet);
}

/* Double precision C entry Point */
kmeansCudaErrorStatus ClosestCentersD(const int m0, const int nRowsA, const int nColsA, 
				      const double *A, const int nColsB, const double *B, 
				      double * Bpadded, const double * normRowsOfA_squared,
				      const double * normColsOfB_squared,
				      const int nColsC, double * C, int *Cindices,
				      int * CindicesFinal) {

  bool constantMemSet = false;
  return ClosestCenters<double,DVECTOR,N_UNROLL_DOUBLE>
    (m0,nRowsA,nColsA,false,A,nColsB,B,Bpadded,normRowsOfA_squared,
     normColsOfB_squared,nColsC,C,Cindices,CindicesFinal, constantMemSet);
}



#if 0
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
#endif      
