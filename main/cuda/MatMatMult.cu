#include "KmeansCudaKernels.h"

#include "multiply.hcu"

template<class TYPE, class VTYPE, const int N_UNROLL, const int DELTA>
__global__ void _dev_MatMatMult(const TYPE * __restrict__ A,
				const TYPE * __restrict__ B,
				TYPE * __restrict__ C) {

  __shared__ VTYPE Ashmem[TILESIZEY][TILESIZEY];
  __shared__ VTYPE Bshmem[TILESIZEY][TILESIZEX];

  int r = blockIdx.y*N_UNROLL*TILESIZE + threadIdx.y;
  int c = blockIdx.x*N_UNROLL*TILESIZE + threadIdx.x;
  VTYPE Creg[N_UNROLL];
  for (int n=0; n<N_UNROLL; ++n)
	zero<VTYPE>(Creg[n]);

  /* matrix matrix multiply */
  TYPE * a = const_cast<TYPE *>(A) + r*dev_nColsA + threadIdx.x;
  TYPE * b = const_cast<TYPE *>(B) + c+threadIdx.y*dev_nColsB;
  multiply<TYPE,VTYPE,N_UNROLL,DELTA>(a, b, Creg, Ashmem, Bshmem);

  /* write the results */
  for (int n=0; n<N_UNROLL; ++n) {
    _dev_writeResults<TYPE,VTYPE>(dev_nRowsA,dev_nColsB,r,c,Creg[n],C);
    r+=TILESIZE;
  }
}



template<class TYPE, class VTYPE, const int N_UNROLL, const int VLENGTH>
__launch_bounds__(64,16)
__global__ void _dev_MatMatMult2(const TYPE * __restrict__ A,
				 const TYPE * __restrict__ B,
				 TYPE * __restrict__ C) {

  __shared__ VTYPE Ashmem[TILESIZE][TILESIZE];
  int r = blockIdx.y*VLENGTH*TILESIZE;
  int c = blockIdx.x*N_UNROLL*TILESIZE;
  int tidx = threadIdx.x + threadIdx.y*blockDim.x;
  
  //VTYPE Creg[TILESIZE] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
  VTYPE Creg[TILESIZE];
  for (int i=0; i<TILESIZE; ++i) {
    Creg[i].x=0;
    Creg[i].y=0;
  }
	 
  TYPE * a = const_cast<TYPE *>(A) + r*dev_nColsA + threadIdx.x;
  TYPE * b = const_cast<TYPE *>(B) + c + tidx;

  multiply2<TYPE, VTYPE, N_UNROLL>(tidx, a, b, &(Creg[0]), Ashmem);
  
  /* write the results */
  for (int i=0; i<TILESIZE; ++i) {
    if (r+i<dev_nRowsA && c+tidx<dev_nColsB) {
      C[(r+i)*dev_nColsB + c + tidx] = Creg[i].x;
    }
    if (r+i+TILESIZE<dev_nRowsA && c+tidx<dev_nColsB) {
      C[(r+i+TILESIZE)*dev_nColsB + c + tidx] = Creg[i].y;
    }
  }
}



/* Generic Templated interface to calling the CUDA kernel */
template<class TYPE, class VTYPE,  const int N_UNROLL>
kmeansCudaErrorStatus MatMatMult(const int nRowsA, const int nColsA, const TYPE *A, 
				 const int nColsB, const TYPE *B, TYPE *C) {
  
  try {
#if 1
    const int nBy = (nRowsA+N_UNROLL*TILESIZE-1)/(N_UNROLL*TILESIZE);
    const int nBx = (nColsB+N_UNROLL*TILESIZE-1)/(N_UNROLL*TILESIZE);
    int N = (nColsA+TILESIZE-1)/(TILESIZE);
    int delta = nColsA-(N-1)*TILESIZE;
    N-=1;
    dim3 grid = dim3(nBx, nBy);
    dim3 block = dim3(TILESIZE, TILESIZE);

    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_N, &N, sizeof(int), 0),ERROR_MATMATMULT);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_nRowsA, &nRowsA, sizeof(int)),ERROR_MATMATMULT);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_nColsA, &nColsA, sizeof(int)),ERROR_MATMATMULT);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_nRowsB, &nColsA, sizeof(int)),ERROR_MATMATMULT);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_nColsB, &nColsB, sizeof(int)),ERROR_MATMATMULT);
    int astride = TILESIZE*nColsA;
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(ASTRIDE, &astride, sizeof(int)),ERROR_MATMATMULT);

    CUDA_SAFE_CALL(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte),ERROR_MATMATMULT);

    if (delta==1) _dev_MatMatMult<TYPE,VTYPE,N_UNROLL,1><<<grid,block>>>(A,B,C);
    if (delta==2) _dev_MatMatMult<TYPE,VTYPE,N_UNROLL,2><<<grid,block>>>(A,B,C);
    if (delta==3) _dev_MatMatMult<TYPE,VTYPE,N_UNROLL,3><<<grid,block>>>(A,B,C);
    if (delta==4) _dev_MatMatMult<TYPE,VTYPE,N_UNROLL,4><<<grid,block>>>(A,B,C);
    if (delta==5) _dev_MatMatMult<TYPE,VTYPE,N_UNROLL,5><<<grid,block>>>(A,B,C);
    if (delta==6) _dev_MatMatMult<TYPE,VTYPE,N_UNROLL,6><<<grid,block>>>(A,B,C);
    if (delta==7) _dev_MatMatMult<TYPE,VTYPE,N_UNROLL,7><<<grid,block>>>(A,B,C);
    if (delta==8) _dev_MatMatMult<TYPE,VTYPE,N_UNROLL,8><<<grid,block>>>(A,B,C);
    if (delta==9) _dev_MatMatMult<TYPE,VTYPE,N_UNROLL,9><<<grid,block>>>(A,B,C);
    if (delta==10) _dev_MatMatMult<TYPE,VTYPE,N_UNROLL,10><<<grid,block>>>(A,B,C);
    if (delta==11) _dev_MatMatMult<TYPE,VTYPE,N_UNROLL,11><<<grid,block>>>(A,B,C);
    if (delta==12) _dev_MatMatMult<TYPE,VTYPE,N_UNROLL,12><<<grid,block>>>(A,B,C);
    if (delta==13) _dev_MatMatMult<TYPE,VTYPE,N_UNROLL,13><<<grid,block>>>(A,B,C);
    if (delta==14) _dev_MatMatMult<TYPE,VTYPE,N_UNROLL,14><<<grid,block>>>(A,B,C);
    if (delta==15) _dev_MatMatMult<TYPE,VTYPE,N_UNROLL,15><<<grid,block>>>(A,B,C);
    if (delta==16) _dev_MatMatMult<TYPE,VTYPE,N_UNROLL,16><<<grid,block>>>(A,B,C);
#else
    const int nBy = (nRowsA+2*TILESIZE-1)/(2*TILESIZE);
    const int nBx = (nColsB+N_UNROLL*TILESIZE-1)/(N_UNROLL*TILESIZE);
    int N = (nColsA+TILESIZE-1)/(TILESIZE);
    int delta = nColsA-(N-1)*TILESIZE;
    N-=1;

    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_N, &N, sizeof(int), 0),ERROR_MATMULT);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_nRowsA, &nRowsA, sizeof(int)),ERROR_MATMULT);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_nColsA, &nColsA, sizeof(int)),ERROR_MATMULT);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_nRowsB, &nColsA, sizeof(int)),ERROR_MATMULT);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_nColsB, &nColsB, sizeof(int)),ERROR_MATMULT);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_DELTA, &delta, sizeof(int)),ERROR_MATMULT);

    dim3 grid = dim3(nBx, nBy);
    dim3 block = dim3(TILESIZE, 4);
    CUDA_SAFE_CALL(cudaFuncSetCacheConfig(_dev_MatMatMult2<TYPE,float2,4,2>, 
					  cudaFuncCachePreferL1),ERROR_MATMATMULT);
    CUDA_SAFE_CALL(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte),ERROR_MATMATMULT);
    _dev_MatMatMult2<TYPE,float2,4,2><<<grid,block>>>(A,B,C);
#endif
    CUDA_SAFE_CALL(cudaGetLastError(),ERROR_MATMATMULT);
  } catch (...) {
    return ERROR_MATMATMULT;
  }
  return NO_ERROR;
}

/* Single precision C entry Point */
kmeansCudaErrorStatus MatMatMultF(const int nRowsA, const int nColsA, const float *A, 
				  const int nColsB, const float *B, float *C) {
  return MatMatMult<float,FVECTOR,N_UNROLL_FLOAT>(nRowsA,nColsA,A,nColsB,B,C);
}

/* Double precision C entry Point */
kmeansCudaErrorStatus MatMatMultD(const int nRowsA, const int nColsA, const double *A, 
				  const int nColsB, const double *B, double *C) {
  return MatMatMult<double,DVECTOR,N_UNROLL_DOUBLE>(nRowsA,nColsA,A,nColsB,B,C);
}
