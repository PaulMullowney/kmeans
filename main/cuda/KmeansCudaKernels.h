#if !defined(KMEANS_CUDA_KERNELS_H_)
#define KMEANS_CUDA_KERNELS_H_

#ifndef KMEANS_CUDA_KERNELS_API
#ifdef _WIN32
#define KMEANS_CUDA_KERNELS_API __stdcall
#else
#define KMEANS_CUDA_KERNELS_API
#endif
#endif

#ifndef DllExport
#ifdef _WIN32
#define DllExport __declspec(dllexport) 
#else
#define DllExport
#endif
#endif

#include "driver_types.h"
#include "channel_descriptor.h"
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "math_constants.h"
#include "vector_types.h"
#include "vector_functions.h"
#include "builtin_types.h"
#include "host_defines.h"


struct __align__(8) float6
{
  float a, b, c, d, e, f;
};

__inline__ __device__ float6 make_float6(float a, float b, float c, float d, float e, float f) {
  float6 y;
  y.a = a;
  y.b = b;
  y.c = c;
  y.d = d;
  y.e = e;
  y.f = f;
  return y;
}

struct __align__(8) int6
{
  int a, b, c, d, e, f;
};

__inline__ __device__ int6 make_int6(int a, int b, int c, int d, int e, int f) {
  int6 y;
  y.a = a;
  y.b = b;
  y.c = c;
  y.d = d;
  y.e = e;
  y.f = f;
  return y;
}


struct __align__(16) float8
{
  float a, b, c, d, e, f, g, h;
};

__inline__ __device__ float8 make_float8(float a, float b, float c, float d, 
					 float e, float f, float g, float h) {
  float8 y;
  y.a = a;
  y.b = b;
  y.c = c;
  y.d = d;
  y.e = e;
  y.f = f;
  y.g = g;
  y.h = h;
  return y;
}

struct __align__(16) int8
{
  int a, b, c, d, e, f, g, h;
};

__inline__ __device__ int8 make_int8(int a, int b, int c, int d,
				     int e, int f, int g, int h) {
  int8 y;
  y.a = a;
  y.b = b;
  y.c = c;
  y.d = d;
  y.e = e;
  y.f = f;
  y.g = g;
  y.h = h;
  return y;
}

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <float.h>
#include <math.h>

/* tile size used in matrix matrix multiplies */
#define TILESIZE 16
#define TILESIZEX 17
#define TILESIZEY 16

/* unrolling parameters based on precision */
#if 0
#define N_BLOCKS 2
#define N_UNROLL_FLOAT 6
typedef float6 FVECTOR;
//#define N_UNROLL_FLOAT 8
//typedef float8 FVECTOR;
#else
#define N_BLOCKS 3
#define N_UNROLL_FLOAT 4
typedef float4 FVECTOR;
#endif

#define N_UNROLL_DOUBLE 2
typedef double2 DVECTOR;

inline float getComputeCapability() {
  cudaDeviceProp prop;
  int device;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop,device);
  return (float)(prop.major + 0.1f*prop.minor);
}

inline int getMaxConcurrentBlocks() {
  cudaDeviceProp prop;
  int device;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop,device);
  if (getComputeCapability()>=5.0)
    return prop.multiProcessorCount*8;
  else if (getComputeCapability()>=3.0)
    return prop.multiProcessorCount*16;
  else
    return prop.multiProcessorCount*8;
}

__constant__ __device__ int dev_N;
__constant__ __device__ int dev_DELTA;
__constant__ __device__ int dev_nRowsAPadded;
__constant__ __device__ int dev_nRowsA;
__constant__ __device__ int dev_nColsA;
__constant__ __device__ int dev_nRowsB;
__constant__ __device__ int dev_nColsB;
__constant__ __device__ int dev_nColsBPadded;
__constant__ __device__ int dev_nColsC;
__constant__ __device__ int ASTRIDE;
__constant__ __device__ int BSTRIDE;

/** 
 * @file 
 *
 * This is the C API to the all the Kmeans Cuda Kernels
 *
 */ 

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */
  
  enum kmeansCudaErrorStatus {
    NO_ERROR = 0, /**< no error */
    ERROR_MATMATMULT = 1, /**< error in MatMatMult */
    ERROR_ROWTRANSFORMMINIMUM = 2, /**< error in RowTransformMinimum */
    ERROR_CLOSESTCENTERS = 3, /**< error in ClosestCenters */
    ERROR_CLUSTERCENTERS = 4, /**< error in ClusterCenters */
    ERROR_COMPACTNESS = 5, /**< error in Compactness */
    ERROR_ROWNORMALIZE = 6, /**< error in RowNormalize */
    ERROR_COLNORMALIZE = 7, /**< error in ColNormalize */
  };

  /**
   * @brief Computes the BLAS 3 product A*B = C on the GPU for
   *  single (float) precision input.
   *
   * @param nRowsA the number of rows in A
   * @param nColsA the number of columns in A
   * @param A a pointer to the left hand side matrix (C ordering)
   * @param nColsC the number of rows in C
   * @param B a pointer to the right hand side matrix (C ordering)
   * @param C a pointer to the result matrix (C ordering)
   *
   * @return an error status upon return
   */
  DllExport kmeansCudaErrorStatus MatMatMultF(const int nRowsA, const int nColsA, const float *A, 
					      const int nColsB, const float *B, float *C);

  /**
   * @brief Computes the BLAS 3 product A*B = C on the GPU for
   *  double precision input.
   *
   * @param nRowsA the number of rows in A
   * @param nColsA the number of columns in A
   * @param A a pointer to the left hand side matrix (C ordering)
   * @param nColsC the number of rows in C
   * @param B a pointer to the right hand side matrix (C ordering)
   * @param C a pointer to the result matrix (C ordering)
   *
   * @return an error status upon return
   */
  DllExport kmeansCudaErrorStatus MatMatMultD(const int nRowsA, const int nColsA, const double *A, 
				    const int nColsB, const double *B, double *C);

  /**
   * @brief Computes the BLAS 3 product A*B = C on the GPU for
   *  and then does a partial reduction on the output of the rows
   *  (single precision input/output).
   *
   * @param m0 the starting data point in a compute tile
   * @param nRowsA the number of rows in A (in a compute tile)
   * @param nColsA the number of columns in A
   * @param A a pointer to the left hand side matrix (C ordering)
   * @param nColsC the number of rows in C
   * @param B a pointer to the right hand side matrix (C ordering)
   * @param B a pointer to the padded right hand side matrix (C ordering)
   * @param normRowsOfA_squared L2 norm squared of the rows of A
   * @param normColsOfB_squared L2 norm squared of the columns of B
   * @param the number of columns in the result data structure
   * @param C a pointer to the result
   * @param Cindices a pointer to the index associated with the partial result
   * @param CindicesFinal a pointer to the index associated with the final result
   *
   * @return an error status upon return
   */
  DllExport kmeansCudaErrorStatus ClosestCentersF(const int m0, const int nRowsA, 
						  const int nColsA, const float *A, 
						  const int nColsB, const float *B, float *Bpadded,
						  const float * normRowsOfA_squared,
						  const float * normColsOfB_squared,
						  const int nColsC, float * C, int *Cindices,
						  int * CindicesFinal);

  /**
   * @brief Computes the BLAS 3 product A*B = C on the GPU 
   *  and then does a partial reduction on the output of the rows
   *  (double precision input/output).
   *
   * @param m0 the starting data point in a compute tile
   * @param nRowsA the number of rows in A (in a compute tile)
   * @param nColsA the number of columns in A
   * @param A a pointer to the left hand side matrix (C ordering)
   * @param nColsC the number of rows in C
   * @param B a pointer to the right hand side matrix (C ordering)
   * @param B a pointer to the padded right hand side matrix (C ordering)
   * @param normRowsOfA_squared L2 norm squared of the rows of A
   * @param normColsOfB_squared L2 norm squared of the columns of B
   * @param the number of columns in the result data structure
   * @param C a pointer to the result
   * @param Cindices a pointer to the index associated with the result
   * @param CindicesFinal a pointer to the index associated with the final result
   *
   * @return an error status upon return
   */
  DllExport kmeansCudaErrorStatus ClosestCentersD(const int m0, const int nRowsA, 
						  const int nColsA, const double *A, 
						  const int nColsB, const double *B, double *Bpadded,
						  const double * normRowsOfA_squared,
						  const double * normColsOfB_squared,
						  const int nColsC, double * C, int *Cindices,
						  int * CindicesFinal);

  /**
   * @brief In a given row i, computes
   *  min over j of : normRowsOfA_squared_i + normColsOfB_squared_j - 2*input_i,j
   *  and stores the resulting index in  the output (single precision input).
   *
   * @param m0 the starting data point in a compute tile
   * @param m the number of rows in the matrix
   * @param n the dimensionality of each data instance
   * @param normRowsOfA_squared the L2 norm squared of the input matrix
   * @param normColsOfB_squared the L2 norm squared of the B matrix 
   * @param input the input matrix
   * @param output the index of the smallest value per row
   *
   * @return an error status upon return
   */
  DllExport kmeansCudaErrorStatus rowTransformMinimumF(const int m0, const int m, const int n, 
						       const float * normRowsOfA_squared, 
						       const float * normColsOfB_squared, 
						       const float * input, int * output);

  /**
   * @brief In a given row i, computes
   *  min over j of : normRowsOfA_squared_i + normColsOfB_squared_j - 2*input_i,j
   *  and stores the resulting index in  the output (double precision input).
   *
   * @param m0 the starting data point in a compute tile
   * @param m the number of rows in the matrix
   * @param n the dimensionality of each data instance
   * @param normRowsOfA_squared the L2 norm squared of the input matrix
   * @param normColsOfB_squared the L2 norm squared of the B matrix 
   * @param input the input matrix
   * @param output the index of the smallest value per row
   *
   * @return an error status upon return
   */
  DllExport kmeansCudaErrorStatus rowTransformMinimumD(const int m0, const int m, const int n,
						       const double * normRowsOfA_squared, 
						       const double * normColsOfB_squared, 
						       const double * input, int * output);


  /**
   * @brief computes the new cluster centers, given a index vector denoting
   *  the closest center for a given row vector. Single precision input/output.
   *
   * @param m0 the starting data point in a compute tile
   * @param m the number of data in a compute tile
   * @param n the dimensionality of each data instance
   * @param k the number of cluster centers
   * @param lastTile if true (we perform the final reduction to get the cluster centers)
   *  on the last tile.
   * @param data the input data. A C-order matrix of dimension m x n.
   * @param indices a vector of length n storing the index of the closest center
   * @param centers_large a temporary data structure used in the reduction
   * @param counts_large a temporary data structure used in the reduction
   * @param centers the new cluster centers. a data structure of size n x k
   *  where each column represents a cluster center. C-ordered.
   * @param counts a vector of length k. Entry i denotes the number of data elements
   *  which contribute to new cluster center i.
   *
   * @return an error status upon return
   */
  DllExport kmeansCudaErrorStatus ClusterCentersF(const int m0, const int m, const int n,
						  const int k, const bool lastTile,
						  const float * data, const int * indices,
						  float * centers_large, int * counts_large,
						  float * centers, int * counts);


  /**
   * @brief computes the new cluster centers, given a index vector denoting
   *  the closest center for a given row vector. Double precision input/output.
   *
   * @param m0 the starting data point in a compute tile
   * @param m the number of data in a compute tile
   * @param n the dimensionality of each data instance
   * @param k the number of cluster centers
   * @param lastTile if true (we perform the final reduction to get the cluster centers)
   *  on the last tile.
   * @param data the input data. A C-order matrix of dimension m x n.
   * @param indices a vector of length n storing the index of the closest center
   * @param centers_large a temporary data structure used in the reduction
   * @param counts_large a temporary data structure used in the reduction
   * @param centers the new cluster centers. a data structure of size n x k
   *  where each column represents a cluster center. C-ordered.
   * @param counts a vector of length k. Entry i denotes the number of data elements
   *  which contribute to new cluster center i.
   *
   * @return an error status upon return
   */
  DllExport kmeansCudaErrorStatus ClusterCentersD(const int m0, const int m, const int n,
						  const int k, const bool lastTile,
						  const double * data, const int * indices,
						  double * centers_large, int * counts_large,
						  double * centers, int * counts);


  /**
   * @brief computes the compactness based on the data, the new cluster centers, the 
   *  index of the closest center. Single precision input/output.
   *
   * @param m0 the starting data point in a compute tile
   * @param m the number of data in a compute tile
   * @param lastTile whether or not this is the last tile
   * @param n the dimensionality of each data instance
   * @param k the number of cluster centers
   * @param data the input data. A C-order matrix of dimension m x n.
   * @param indices a vector of length n storing the index of the closest center
   * @param centers the cluster centers. a data structure of size n x k
   *  where each column represents a cluster center. C-ordered.
   * @param compactness the compactness of the data.
   *
   * @return an error status upon return
   */
  DllExport kmeansCudaErrorStatus CompactnessF(const int m0, const int m, 
					       const bool lastTile, const int n, const int k,
					       const float * data, const int * indices,
					       const float * centers, float * compactness);
  
  /**
   * @brief computes the compactness based on the data, the new cluster centers, the 
   *  index of the closest center. Double precision input/output.
   *
   * @param m0 the starting data point in a compute tile
   * @param m the number of data in a compute tile
   * @param lastTile whether or not this is the last tile
   * @param n the dimensionality of each data instance
   * @param k the number of cluster centers
   * @param data the input data. A C-order matrix of dimension m x n.
   * @param indices a vector of length n storing the index of the closest center
   * @param centers the cluster centers. a data structure of size n x k
   *  where each column represents a cluster center. C-ordered.
   * @param compactness the compactness of the data.
   *
   * @return an error status upon return
   */
  DllExport kmeansCudaErrorStatus CompactnessD(const int m0, const int m, 
					       const bool lastTile, const int n, const int k,
					       const double * data, const int * indices,
					       const double * centers, double * compactness);


  /**
   * @brief Compute the L2 norm (squared) of each row. Single precision input/output.
   *
   * @param m0 the starting data point in a compute tile
   * @param m the number of data in a compute tile
   * @param n the dimensionality of each data instance
   * @param input the input matrix
   * @param output the L2 norm squared of each row
   *
   * @return an error status upon return
   */
  DllExport kmeansCudaErrorStatus rowNormalizeF(const int m0, const int m, const int n, 
						const float * __restrict__ input, 
						float * __restrict__ output);
  
  /**
   * @brief Compute the L2 norm (squared) of each row. Double precision input/output.
   *
   * @param m0 the starting data point in a compute tile
   * @param m the number of data in a compute tile
   * @param n the dimensionality of each data instance
   * @param input the input matrix
   * @param output the L2 norm squared of each row
   *
   * @return an error status upon return
   */
  DllExport kmeansCudaErrorStatus rowNormalizeD(const int m0, const int m, const int n, 
						const double * __restrict__ input, 
						double * __restrict__ output);


  /**
   * @brief Compute the L2 norm (squared) of each col. Single precision input/output.
   *
   * @param m the number of rows in the matrix
   * @param n the number of columns in the matrix
   * @param input the input matrix
   * @param output the L2 norm squared of each col
   *
   * @return an error status upon return
   */
  DllExport kmeansCudaErrorStatus colNormalizeF(const int m, const int n, 
						const float * __restrict__ input, 
						float * __restrict__ output);
  
  /**
   * @brief Compute the L2 norm (squared) of each col. Double precision input/output.
   *
   * @param m the number of rows in the matrix
   * @param n the number of columns in the matrix
   * @param input the input matrix
   * @param output the L2 norm squared of each col
   *
   * @return an error status upon return
   */
  DllExport kmeansCudaErrorStatus colNormalizeD(const int m, const int n,
						const double * __restrict__ input, 
						double * __restrict__ output);

#if defined(__cplusplus)
}
#endif /* __cplusplus */                         


/**
 * @brief Computes the BLAS 3 product A*B = C on the GPU for
 *  and then does a partial reduction on the output of the rows
 *  Arbitrary TYPE.
 *
 * @param m0 the starting data point in a compute tile
 * @param nRowsA the number of rows in A
 * @param nColsA the number of columns in A
 * @param isStriped whether or not the supplied matrix is striped
 * @param A a pointer to the left hand side matrix or the Transpose (C ordering)
 * @param nColsC the number of rows in C
 * @param B a pointer to the right hand side matrix (C ordering)
 * @param B a pointer to the padded right hand side matrix (C ordering)
 * @param normRowsOfA_squared L2 norm squared of the rows of A
 * @param normColsOfB_squared L2 norm squared of the columns of B
 * @param the number of columns in the result data structure
 * @param C a pointer to the result
 * @param Cindices a pointer to the index associated with the partial result
 * @param CindicesFinal a pointer to the index associated with the final result
 *
 * @return an error status upon return
 */
template<class TYPE>
DllExport kmeansCudaErrorStatus ClosestCenters(const int m0, const int nRowsA, const int nColsA, 
					       const bool isStriped, const TYPE *A,
					       const int nColsB, const TYPE *B, TYPE *Bpadded, 
					       const TYPE * normRowsOfA_squared,
					       const TYPE * normColsOfB_squared,
					       const int nColsC, TYPE * C, int *Cindices,
					       int * CindicesFinal, bool& constantMemSet);

/**
 * @brief computes the new cluster centers, given a index vector denoting
 *  the closest center for a given row vector. Arbitrary TYPE.
 *
 * @param m0 the starting data point in a compute tile
 * @param m the number of data in a compute tile
 * @param n the dimensionality of each data instance
 * @param k the number of cluster centers
 * @param lastTile if true (we perform the final reduction to get the cluster centers)
 *  on the last tile.
 * @param data the input data. A C-order matrix of dimension m x n.
 * @param indices a vector of length n storing the index of the closest center
 * @param centers_large a temporary data structure used in the reduction
 * @param counts_large a temporary data structure used in the reduction
 * @param centers the new cluster centers. a data structure of size n x k
 *  where each column represents a cluster center. C-ordered.
 * @param counts a vector of length k. Entry i denotes the number of data elements
 *  which contribute to new cluster center i.
 *
 * @return an error status upon return
 */
template<class TYPE>
DllExport kmeansCudaErrorStatus ClusterCenters(const int m0, const int m, const int n,
					       const int k, const bool lastTile,
					       const TYPE * data, const int * indices,
					       TYPE * centers_large, int * counts_large,
					       TYPE * centers, int * counts);

/**
 * @brief computes the compactness based on the data, the new cluster centers, the 
 *  index of the closest center. Arbitrary TYPE.
 *
 * @param m0 the starting data point in a compute tile
 * @param m the number of data in a compute tile
 * @param lastTile whether or not this is the last tile
 * @param n the dimensionality of each data instance
 * @param k the number of cluster centers
 * @param data the input data. A C-order matrix of dimension m x n.
 * @param indices a vector of length n storing the index of the closest center
 * @param centers the cluster centers. a data structure of size n x k
 *  where each column represents a cluster center. C-ordered.
 * @param compactness the compactness of the data.
 *
 * @return an error status upon return
 */
template<class TYPE>
DllExport kmeansCudaErrorStatus Compactness(const int m0, const int m, 
					    const bool lastTile, const int n, const int k,
					    const TYPE * data, const int * indices,
					    const TYPE * centers, TYPE * compactness);

/**
 * @brief Compute the L2 norm (squared) of each row. Arbitrary TYPE.
 *
 * @param m0 the starting data point in a compute tile
 * @param m the number of data in a compute tile
 * @param n the dimensionality of each data instance
 * @param input the input matrix
 * @param output the L2 norm squared of each row
 *
 * @return an error status upon return
 */
template<class TYPE>
DllExport kmeansCudaErrorStatus rowNormalize(const int m0, const int m, const int n,
					     const TYPE * __restrict__ input, 
					     TYPE * __restrict__ output);

/**
 * @brief Compute the L2 norm (squared) of each col. Arbitrary TYPE.
 *
 * @param m the number of rows in the matrix
 * @param n the number of columns in the matrix
 * @param input the input matrix
 * @param output the L2 norm squared of each col
 *
 * @return an error status upon return
 */
template<class TYPE>
DllExport kmeansCudaErrorStatus colNormalize(const int m, const int n,
					     const TYPE * __restrict__ input, 
					     TYPE * __restrict__ output);



#define CUDA_SAFE_CALL(err,kmeans_err)  __cudaSafeCall(err, kmeans_err, __FILE__, __LINE__)
inline void __cudaSafeCall(cudaError_t err, kmeansCudaErrorStatus kmeansError, const char *file, const int line) {
  if(cudaSuccess != err) {
    printf("%s(%i) : cudaSafeCall() Runtime API error : %s.\n",
           file, line, cudaGetErrorString(err) );
    throw kmeansError;
  }
}

#endif /* !defined(KMEANS_CUDA_KERNELS_H_) */
