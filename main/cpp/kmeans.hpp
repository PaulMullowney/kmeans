#if !defined(KMEANS_HPP_)
#define KMEANS_HPP_

#ifndef KMEANS_API
#ifdef _WIN32
#define KMEANS_API __stdcall
#else
#define KMEANS_API
#endif
#endif

#include "config.h"
#include <vector>
#include <algorithm>
#include <iostream>
#include <random>
#include <chrono>
#include <array>
#include <string>
#include <string.h>

#ifdef HAVE_MAGMA_H
#include <magma.h>
#endif

using namespace std;

#include "KmeansCudaKernels.h"

#ifndef DllExport
#ifdef _WIN32
#define DllExport __declspec(dllexport) 
#else
#define DllExport
#endif
#endif

#include "cublas_v2.h"

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

  enum kmeansErrorStatus {
    KMEANS_SUCCESS = 0, /**< no error */
    KMEANS_ERROR_BUILDDATA = 1, /**< error in building the data */
    KMEANS_ERROR_INITIALIZE = 2, /**< error in the initialization */
    KMEANS_ERROR_CLOSEST_CENTERS = 3, /**< error in computing the closest centers */
    KMEANS_ERROR_CLUSTER_CENTERS = 4, /**< error in computing the cluster centers */
    KMEANS_ERROR_COMPACTNESS = 5, /**< error in computing the compactness */
    KMEANS_ERROR_TRANSPOSE = 6, /**< error in transpose */
    KMEANS_ERROR_RESET = 7, /**< error in reset */
    KMEANS_ERROR_COMPUTE = 8, /**< error in compute */
    KMEANS_ERROR_GET_CENTERS = 9, /**< error in copying the data back to the host */
    KMEANS_ERROR_TILING = 10, /**< error in the tiling */
    KMEANS_ERROR_COPY_TILE = 11, /**< error copying a tile of data to the device */
    KMEANS_ERROR_CONSTRUCT_MINIBATCH = 12, /**< error in constructing the mini batch */
    KMEANS_NONCONVERGENCE = 13, /**< algorithm did not converge */
  };

  /**
   * @brief get an error string associated with the error code
   *
   * @return a string describing the error
   */
  const char * kmeansGetErrorString(kmeansErrorStatus err) {
    if (err==KMEANS_ERROR_BUILDDATA) return "error in building the data";
    else if (err==KMEANS_ERROR_INITIALIZE) return "error in the initialization";
    else if (err==KMEANS_ERROR_CLOSEST_CENTERS) return "error in computing the closest centers";
    else if (err==KMEANS_ERROR_CLUSTER_CENTERS) return "error in computing the cluster centers";
    else if (err==KMEANS_ERROR_COMPACTNESS) return "error in computing the compactness";
    else if (err==KMEANS_ERROR_TRANSPOSE) return "error in transpose";
    else if (err==KMEANS_ERROR_RESET) return "error in reset";
    else if (err==KMEANS_ERROR_COMPUTE) return "error in compute";
    else if (err==KMEANS_ERROR_GET_CENTERS) return "error in copying the data back to the host";
    else if (err==KMEANS_ERROR_TILING) return "error in the tiling";
    else if (err==KMEANS_ERROR_COPY_TILE) return "error in copy tile";
    else if (err==KMEANS_ERROR_CONSTRUCT_MINIBATCH) return "error in constructing the minibatch";
    else if (err==KMEANS_NONCONVERGENCE) return "algorithm did not converge";
    else return "no error";

  }

  /**
   * @brief this is the public interface for calling the compute kmeans
   * routine. Single precision.
   *
   * @param data the input data (C-ordered)
   * @param m the number of rows in the matrix
   * @param n the number of columns in the matrix (the dimensionality of the data)
   * @param k the number of centers
   * @param criterion the convergence criterion
   * @param maxIters the maximum number of iterations
   * @param numRetries the number of retries for computing the best centers
   * @param initAlgorithm the initialization algorithm
   * @param doFullSGEMM whether or not to do a full matrix matrix multiply (1==CUBLAS, 2==MAGMA)
   * @param matrixFormat format to use for the computation. Normal, Transpose, Striped
   * @param result the buffer for storing the result
   *
   * @return a code detailing error status after completion.
   */
  DllExport kmeansErrorStatus KMEANS_API computeKmeansF(const float * data, const int m, const int n,
							const int k, const float criterion,
							const int maxIters, const int numRetries,
							const int initAlgorithm, const int doFullSGEMM,
							const int matrixFormat, 
							float * result, const float miniBatchFraction=1.0);

  /**
   * @brief this is the public interface for calling the compute kmeans
   * routine. Double precision.
   *
   * @param data the input data (C-ordered)
   * @param m the number of rows in the matrix
   * @param n the number of columns in the matrix (the dimensionality of the data)
   * @param k the number of centers
   * @param criterion the convergence criterion
   * @param maxIters the maximum number of iterations
   * @param numRetries the number of retries for computing the best centers
   * @param initAlgorithm the initialization algorithm
   * @param doFullSGEMM whether or not to do a full matrix matrix multiply (1==CUBLAS, 2==MAGMA)
   * @param matrixFormat format to use for the computation. Normal or Striped
   * @param result the buffer for storing the result
   *
   * @return a code detailing error status after completion.
   */
  DllExport kmeansErrorStatus KMEANS_API computeKmeansD(const double * data, const int m, const int n,
							const int k, const double criterion,
							const int maxIters, const int numRetries,
							const int initAlgorithm, const int doFullSGEMM,
							const int matrixFormat, 
							double * result, const double miniBatchFraction=1.0);

#if defined(__cplusplus)
}
#endif /* __cplusplus */                         

#define CUDA_API_SAFE_CALL(err,kmeans_err)  __cudaAPISafeCall(err, kmeans_err, __FILE__, __LINE__)
inline void __cudaAPISafeCall(cudaError_t err, kmeansErrorStatus kmeansError, const char *file, const int line) {
  if (cudaSuccess != err) {
    printf("%s(%i) : cudaSafeCall() Runtime API error : %s.\n",
	   file, line, cudaGetErrorString(err));
    throw kmeansError;
  }
}


#define CUBLAS_API_SAFE_CALL(err,kmeans_err)  __cublasSafeCall(err, kmeans_err, __FILE__, __LINE__)
inline void __cublasSafeCall(cublasStatus_t err, kmeansErrorStatus kmeansError, const char *file, const int line) {
  if (CUBLAS_STATUS_SUCCESS != err) {
    printf("%s(%i) : cublasSafeCall() Runtime API error : %d.\n",
	   file, line, int(err));
    throw kmeansError;
  }
}


#define START_TIMER(dt,event,kmeans_err) startTimer(dt,event,kmeans_err)
inline void startTimer(float& dt, cudaEvent_t e, kmeansErrorStatus kmeans_err) {
  dt = 0.f;
  CUDA_API_SAFE_CALL(cudaEventRecord(e, 0), kmeans_err);
}

#define STOP_TIMER(dt,start,stop,dtVar,kmeans_err) stopTimer(dt,start,stop,dtVar,kmeans_err)
inline void stopTimer(float& dt, cudaEvent_t start, cudaEvent_t stop, float &dtVar, kmeansErrorStatus kmeans_err) {
    /* stop the timer */
    CUDA_API_SAFE_CALL(cudaEventRecord(stop, 0), kmeans_err);
    CUDA_API_SAFE_CALL(cudaEventSynchronize(stop), kmeans_err);
    CUDA_API_SAFE_CALL(cudaEventElapsedTime(&dt, start, stop), kmeans_err);
    dtVar += ((float).001)*dt;
}


template<class TYPE>
class kmeans {

public:

  /**
   * @brief constructor where the data type is also specified
   */
  kmeans(const int m, const int n, const int k,
	 const TYPE criterion, const int maxIters,
	 const int numRetries, const int initAlgorithm,
	 const int doFullSGEMM, const int matrixFormat,
	 const TYPE miniBatchFraction=1.0);

  /**
   * @brief destructor
   */
  virtual ~kmeans();

  /**
   * @brief return whether or not convergence occurred
   *
   * @return an error status denoting success or failure
   */
  virtual kmeansErrorStatus getConvergenceFlag() {
    return this->relErr <= this->criterion ? KMEANS_SUCCESS : KMEANS_NONCONVERGENCE;
  }

  /**
   * @brief copy the data from device to host
   *
   * @param result host buffer which receives the data from the device
   *
   * @return an error status denoting success or failure
   */
  virtual kmeansErrorStatus getClusterCenters(TYPE * result);

  /**
   * @brief build the device data
   *
   * @param data the input data
   *
   * @return an error status denoting success or failure
   */
  virtual kmeansErrorStatus buildData(const TYPE * data);

  /**
   * @brief initialize the data
   *
   * @param result host buffer which receives the data from the device
   *
   * @return an error status denoting success or failure
   */
  virtual kmeansErrorStatus initialize(TYPE * result);

  /**
   * @brief construct the minibatch
   *
   * @param index the index of the minibatch in the current tile
   *
   * @return an error status denoting success or failure
   */
  virtual kmeansErrorStatus constructMiniBatch(int index);

  /**
   * @brief run the computation
   *
   * @return an error status denoting success or failure
   */
  virtual kmeansErrorStatus compute();

  /**
   * @brief run the computation using the standard approach (i.e.
   *  non minibatch)
   *
   * @return an error status denoting success or failure
   */
  virtual kmeansErrorStatus computeStandard();

  /**
   * @brief run the computation with the mini batch algorithm
   *  when the data is very large
   *
   * @return an error status denoting success or failure
   */
  virtual kmeansErrorStatus computeMiniBatch();

  /**
   * @brief compute closest centers
   *
   * @return an error status denoting success or failure
   */
  virtual kmeansErrorStatus closestCenters();

  /**
   * @brief compute the new cluster centers
   *
   * @return an error status denoting success or failure
   */
  virtual kmeansErrorStatus clusterCenters();

  /**
   * @brief compute the new cluster centers
   *
   * @return an error status denoting success or failure
   */
  virtual kmeansErrorStatus compactness();

  /**
   * @brief compute the tranpose of the cluster centers
   *
   * @return an error status denoting success or failure
   */
  virtual kmeansErrorStatus transpose();

  /**
   * @brief reset the data
   *
   * @return an error status denoting success or failure
   */
  virtual kmeansErrorStatus reset();

  /**
   * @brief compute the tiling. If the matrix and the intermediate
   *  data arrays are too large to fit into device memory, the array
   *  must be tiled. This method computes the tiling.
   *
   * @return an error status denoting success or failure
   */
  virtual kmeansErrorStatus computeTiling();

  /**
   * @brief copy the current tile to the device
   *
   * @param initialize if used in initialize, we use the old style tiling code
   *
   * @return an error status denoting success or failure
   */
  virtual kmeansErrorStatus copyTileToDevice(bool standardApproach=true);

protected:

private:

  /**
   * the current tile being processed.
   */
  int iTile;

  /**
   * whether or not the current tile is the last tile to be processed
   */
  bool lastTile;

  /**
   * whether or not the entire data set is on the device
   */
  bool entireDatasetIsOnDevice;

  /**
   * the number of tiles required to fit the computation in device memory
   */
  int nTiles;

  /**
   * the starting row in the matrix for the ClosestCenter, ClusterCenters and
   * Compactness calculations
   */
  vector<int> mStart;

  /**
   * the number of row to compute for the ClosestCenter, ClusterCenters and
   * Compactness calculations
   */
  vector<int> mTile;

  /**
   * whether or not the constant memory has been set
   */
  bool constantMemSet;

  /**
   * whether or not to use the CUBLAS library for the matrix matrix multiply
   */
  bool useCUBLAS;

  /**
   * whether or not to use the MAGMA library for the matrix matrix multiply
   */
  bool useMAGMA;

  /**
   * whether or not to use the transposed matrix for the matrix matrix multiply
   */
  bool useStriped;

  /**
   * whether or not to use the minibatch version of the algorithm
   */
  bool useMiniBatch;

  /**
   * whether or not to use the minibatch version of the algorithm
   */
  bool smallMiniBatch;

  /**
   * the fraction of the data to use in mini batch
   */
  float miniBatchFraction;

  /**
   * vector of starting points for the minibatch
   */
  vector<int> mMBStart;

  /**
   * vector of booleans which indicates if the minibatch wraps around the data set
   */
  vector<bool> mMBWrapTile;

  /**
   * The number of minibatches per tile
   */
  int num_mb_per_tile;

  /**
   * The number of distinct data points (i.e. the number of rows in the data)
   * for minibatch
   */
  int m_mb;

  /**
   * The number of distinct data points (i.e. the number of rows in the data)
   * rounded up to the next multiple of TILESIZE*N_UNROLL_FLOAT (DOUBLE)
   * for minibatch
   */
  int m_mb_padded;

  /**
   * Index into the array for the starting and end points of the minibatch.
   */
  int m_mb_index;

  /**
   * A device array of size m_mb_added x n containing the mini batch input data
   */
  TYPE * dev_data_mb;

  /**
   * A device array of length m storing the norm of each row (squared) of the input data
   */
  TYPE * dev_data_norm_squared_mb;

  /**
   * handle to the cublas library
   */
  cublasHandle_t handle;

  /**
   * start timer
   */
  cudaEvent_t start;

  /**
   * stop timer
   */
  cudaEvent_t stop;

  /**
   * timer
   */
  float DT;

  /**
   * timer
   */
  float dtBuild;

  /**
   * timer
   */
  float dtRowNormalize;

  /**
   * timer
   */
  float dtInitialize;

  /**
   * timer
   */
  float dtCopy;

  /**
   * timer
   */
  float dtGetCenters;

  /**
   * timer
   */
  float dtConstructMiniBatch;

  /**
   * timer
   */
  float dtClosestCenters;

  /**
   * timer
   */
  float dtClusterCenters;

  /**
   * timer
   */
  float dtCompactness;

  /**
   * timer
   */
  float dtTranspose;

  /**
   * timer
   */
  float dtReset;

  /**
   * The convergence criterion
   */
  TYPE criterion;

  /**
   * The maximum number of iterations
   */
  int maxIters;

  /**
   * The number of retries used to get the best estimate of the cluster centers
   */
  int numRetries;

  /**
   * The number of compute data points (i.e. the number of rows in the data)
   */
  int m;

  /**
   * The padded number of compute data points (i.e. the number of rows in the data)
   * rounded up to the next multiple of TILESIZE*N_UNROLL_FLOAT (DOUBLE)
   */
  int m_padded;

  /**
   * The dimensionality of the data points (i.e. the number of columns in the data)
   */
  int n;

  /**
   * The number of cluster centers
   */
  int k;

  /**
   * The factor for the partial reduction : 64 for float, 32 for double
   */
  int factor;

  /**
   * The number of columns in the partial reduction data structure
   */
  int p;

  /**
   * A device array of size m x n containing the input data
   */
  TYPE * host_data;

  /**
   * A device array of size m x n containing the input data
   */
  TYPE * dev_data;

  /**
   * A device array of size m x n containing the input data
   */
  TYPE * dev_data_aux;

  /**
   * whether or not the striped matrix has been created
   */
  bool hasStripedMatrix;

  /**
   * pointer to the matrix matrix multiply result
   */
  TYPE * dev_mmMult;

  /**
   * A device array of length m storing the norm of each row (squared) of the input data
   */
  TYPE * dev_data_norm_squared;

  /**
   * the result of the partial reduction
   */
  TYPE * dev_partial_reduction;

  /**
   * the indices used in the partial
   */
  int * dev_partial_reduction_indices;

  /**
   * A device array of length m storing the index of the closest center
   */
  int * dev_ccindex;

  /**
   * A device array of size k x n containing the cluster centers (C ordered)
   */
  TYPE * dev_centers;

  /**
   * A device array of size k x n containing the cluster centers (C ordered)
   * Here we pad to the next multiple of N_UNROLL*TILESIZE (in the columns direction) 
   * in order to guarantee alignment
   */
  TYPE * dev_centers_padded;

  /**
   * A device array of size k x n containing the cluster centers (Fortran ordered)
   */
  TYPE * dev_centers_transpose;

  /**
   * A device array of size maxConcurrentBlocks x k x n containing the cluster centers across
   * different device compute blocks. This array is further reduced to compute the actual
   * cluster centers (Fortran ordered).
   */
  TYPE * dev_centers_large;

  /**
   * A device array of length k storing the norm (squared) of each cluster center
   */
  TYPE * dev_centers_norm_squared;

  /**
   * A device array of length k storing the number of data points contributing to each
   * new cluster center
   */
  int * dev_counts;

  /**
   * A device array of length maxConcurrentBlocks x k storing the number of data points contributing
   * to each new cluster center across different device compute blocks. This array is further
   * reduced to compute the actual counts per cluster center.
   */
  int * dev_counts_large;

  /**
   * Compactness score
   */
  TYPE compactnessScore;

  /**
   * Relative Error in the compactness score
   */
  TYPE relErr;

  /**
   * A device array of length k storing the contribution of each (new) cluster center
   * to the total compactness score
   */
  TYPE * dev_compactness;

  /**
   * A host array of length maxConcurrentBlocks storing the contribution of each (new) cluster center
   * to the total compactness score
   */
  TYPE host_compactness;

};

#endif /* !defined(KMEANS_HPP_) */
