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
#define N_UNROLL_FLOAT 4
typedef float4 FVECTOR;

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
__constant__ __device__ int dev_nRowsA;
__constant__ __device__ int dev_nColsA;
__constant__ __device__ int dev_nRowsB;
__constant__ __device__ int dev_nColsB;
__constant__ __device__ int dev_nColsC;
__constant__ __device__ int ASTRIDE;

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
   * @param nRowsA the number of rows in A
   * @param nColsA the number of columns in A
   * @param A a pointer to the left hand side matrix (C ordering)
   * @param nColsC the number of rows in C
   * @param B a pointer to the right hand side matrix (C ordering)
   * @param normRowsOfA_squared L2 norm squared of the rows of A
   * @param normColsOfB_squared L2 norm squared of the columns of B
   * @param the number of columns in the result data structure
   * @param C a pointer to the result
   * @param Cindices a pointer to the index associated with the partial result
   * @param CindicesFinal a pointer to the index associated with the final result
   *
   * @return an error status upon return
   */
  DllExport kmeansCudaErrorStatus ClosestCentersF(const int nRowsA, const int nColsA, const float *A, 
					const int nColsB, const float *B, 
					const float * normRowsOfA_squared,
					const float * normColsOfB_squared,
					const int nColsC, float * C, int *Cindices,
					int * CindicesFinal);

  /**
   * @brief Computes the BLAS 3 product A*B = C on the GPU 
   *  and then does a partial reduction on the output of the rows
   *  (double precision input/output).
   *
   * @param nRowsA the number of rows in A
   * @param nColsA the number of columns in A
   * @param A a pointer to the left hand side matrix (C ordering)
   * @param nColsC the number of rows in C
   * @param B a pointer to the right hand side matrix (C ordering)
   * @param normRowsOfA_squared L2 norm squared of the rows of A
   * @param normColsOfB_squared L2 norm squared of the columns of B
   * @param the number of columns in the result data structure
   * @param C a pointer to the result
   * @param Cindices a pointer to the index associated with the result
   * @param CindicesFinal a pointer to the index associated with the final result
   *
   * @return an error status upon return
   */
  DllExport kmeansCudaErrorStatus ClosestCentersD(const int nRowsA, const int nColsA, const double *A, 
					const int nColsB, const double *B, 
					const double * normRowsOfA_squared,
					const double * normColsOfB_squared,
					const int nColsC, double * C, int *Cindices,
					int * CindicesFinal);

  /**
   * @brief In a given row i, computes
   *  min over j of : normRowsOfA_squared_i + normColsOfB_squared_j - 2*input_i,j
   *  and stores the resulting index in  the output (single precision input).
   *
   * @param n the number of rows in the matrix
   * @param k the number of columns in the matrix
   * @param normRowsOfA_squared the L2 norm squared of the input matrix
   * @param normColsOfB_squared the L2 norm squared of the B matrix 
   * @param input the input matrix
   * @param output the index of the smallest value per row
   *
   * @return an error status upon return
   */
  DllExport kmeansCudaErrorStatus rowTransformMinimumF(const int n, const int k, 
					     const float * normRowsOfA_squared, 
					     const float * normColsOfB_squared, 
					     const float * input, int * output);

  /**
   * @brief In a given row i, computes
   *  min over j of : normRowsOfA_squared_i + normColsOfB_squared_j - 2*input_i,j
   *  and stores the resulting index in  the output (double precision input).
   *
   * @param n the number of rows in the matrix
   * @param k the number of columns in the matrix
   * @param normRowsOfA_squared the L2 norm squared of the input matrix
   * @param normColsOfB_squared the L2 norm squared of the B matrix 
   * @param input the input matrix
   * @param output the index of the smallest value per row
   *
   * @return an error status upon return
   */
  DllExport kmeansCudaErrorStatus rowTransformMinimumD(const int n, const int k,
					     const double * normRowsOfA_squared, 
					     const double * normColsOfB_squared, 
					     const double * input, int * output);


  /**
   * @brief computes the new cluster centers, given a index vector denoting
   *  the closest center for a given row vector. Single precision input/output.
   *
   * @param m the number of data
   * @param n the dimensionality of each data instance
   * @param k the number of cluster centers
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
  DllExport kmeansCudaErrorStatus ClusterCentersF(const int m, const int n, const int k,
					const float * data, const int * indices,
					float * centers_large, int * counts_large,
					float * centers, int * counts);


  /**
   * @brief computes the new cluster centers, given a index vector denoting
   *  the closest center for a given row vector. Double precision input/output.
   *
   * @param m the number of data
   * @param n the dimensionality of each data instance
   * @param k the number of cluster centers
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
  DllExport kmeansCudaErrorStatus ClusterCentersD(const int m, const int n, const int k,
					const double * data, const int * indices,
					double * centers_large, int * counts_large,
					double * centers, int * counts);


  /**
   * @brief computes the compactness based on the data, the new cluster centers, the 
   *  index of the closest center. Single precision input/output.
   *
   * @param m the number of data
   * @param n the dimensionality of each data instance
   * @param k the number of cluster centers
   * @param data the input data. A C-order matrix of dimension m x n.
   * @param indices a vector of length n storing the index of the closest center
   * @param centers the cluster centers. a data structure of size n x k
   *  where each column represents a cluster center. C-ordered.
   * @param compactness the compactness of the data.
   * @param compactness_cpu the compactness of the data ... on the cpu
   *
   * @return an error status upon return
   */
  DllExport kmeansCudaErrorStatus CompactnessF(const int m, const int n, const int k,
				     const float * data, const int * indices,
				     const float * centers, float * compactness,
				     float * compactness_cpu);
  
  /**
   * @brief computes the compactness based on the data, the new cluster centers, the 
   *  index of the closest center. Double precision input/output.
   *
   * @param m the number of data
   * @param n the dimensionality of each data instance
   * @param k the number of cluster centers
   * @param data the input data. A C-order matrix of dimension m x n.
   * @param indices a vector of length n storing the index of the closest center
   * @param centers the cluster centers. a data structure of size n x k
   *  where each column represents a cluster center. C-ordered.
   * @param compactness the compactness of the data.
   * @param compactness_cpu the compactness of the data ... on the cpu
   *
   * @return an error status upon return
   */
  DllExport kmeansCudaErrorStatus CompactnessD(const int m, const int n, const int k,
				     const double * data, const int * indices,
				     const double * centers, double * compactness,
				     double * compactness_cpu);


  /**
   * @brief Compute the L2 norm (squared) of each row. Single precision input/output.
   *
   * @param m the number of rows in the matrix
   * @param n the number of columns in the matrix
   * @param input the input matrix
   * @param output the L2 norm squared of each row
   *
   * @return an error status upon return
   */
	DllExport kmeansCudaErrorStatus rowNormalizeF(const int m, const int n, 
				      const float * __restrict__ input, 
				      float * __restrict__ output);
  
  /**
   * @brief Compute the L2 norm (squared) of each row. Double precision input/output.
   *
   * @param m the number of rows in the matrix
   * @param n the number of columns in the matrix
   * @param input the input matrix
   * @param output the L2 norm squared of each row
   *
   * @return an error status upon return
   */
  DllExport kmeansCudaErrorStatus rowNormalizeD(const int m, const int n, 
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
   * @param nRowsA the number of rows in A
   * @param nColsA the number of columns in A
   * @param A a pointer to the left hand side matrix (C ordering)
   * @param nColsC the number of rows in C
   * @param B a pointer to the right hand side matrix (C ordering)
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
DllExport kmeansCudaErrorStatus ClosestCenters(const int nRowsA, const int nColsA, const TYPE *A,
					       const int nColsB, const TYPE *B, 
					       const TYPE * normRowsOfA_squared,
					       const TYPE * normColsOfB_squared,
					       const int nColsC, TYPE * C, int *Cindices,
					       int * CindicesFinal, bool& constantMemSet);

/**
 * @brief computes the new cluster centers, given a index vector denoting
 *  the closest center for a given row vector. Arbitrary TYPE.
 *
 * @param m the number of data
 * @param n the dimensionality of each data instance
 * @param k the number of cluster centers
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
DllExport kmeansCudaErrorStatus ClusterCenters(const int m, const int n, const int k,
					       const TYPE * data, const int * indices,
					       TYPE * centers_large, int * counts_large,
					       TYPE * centers, int * counts);

/**
 * @brief computes the compactness based on the data, the new cluster centers, the 
 *  index of the closest center. Arbitrary TYPE.
 *
 * @param m the number of data
 * @param n the dimensionality of each data instance
 * @param k the number of cluster centers
 * @param data the input data. A C-order matrix of dimension m x n.
 * @param indices a vector of length n storing the index of the closest center
 * @param centers the cluster centers. a data structure of size n x k
 *  where each column represents a cluster center. C-ordered.
 * @param compactness the compactness of the data.
 * @param compactness_cpu the compactness of the data ... on the cpu
 *
 * @return an error status upon return
 */
template<class TYPE>
DllExport kmeansCudaErrorStatus Compactness(const int m, const int n, const int k,
					    const TYPE * data, const int * indices,
					    const TYPE * centers, TYPE * compactness,
					    TYPE * compactness_cpu);

/**
 * @brief Compute the L2 norm (squared) of each row. Arbitrary TYPE.
 *
 * @param m the number of rows in the matrix
 * @param n the number of columns in the matrix
 * @param input the input matrix
 * @param output the L2 norm squared of each row
 *
 * @return an error status upon return
 */
template<class TYPE>
DllExport kmeansCudaErrorStatus rowNormalize(const int m, const int n,
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
