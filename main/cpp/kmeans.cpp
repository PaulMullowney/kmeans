#include "kmeans.hpp"

template<class TYPE>
kmeans<TYPE>::kmeans(const int m, const int n, const int k,
		     const TYPE criterion, const int maxIters,
		     const int numRetries, const int initAlgorithm,
		     const int useCUBLAS) {

  /* set the key meta data from the input */
  this->m = m;
  this->n = n;
  this->k = k;

  if (sizeof(TYPE) == 4)
    this->factor = TILESIZE*N_UNROLL_FLOAT;
  else
    this->factor = TILESIZE*N_UNROLL_DOUBLE;
  
  this->m_padded = ((this->m + this->factor - 1) / this->factor)*this->factor;
  this->p = (this->k + this->factor - 1) / this->factor;
  
  this->criterion = criterion;
  this->maxIters = maxIters;
  this->numRetries = numRetries;

  /* set the convergence meta data */
  this->compactnessScore = 0.;
  this->relErr = 1.0;
  
  /* calculate the maximum number of blocks give a particular device */
  this->host_compactness.resize(getMaxConcurrentBlocks());
  
  this->handle = NULL;
  cout << "useCUBLAS=" << useCUBLAS << endl;
  if (useCUBLAS) this->useCUBLAS = true;
  else this->useCUBLAS = false;
  
  this->constantMemSet = false;
  
  /* timing data */
  this->dtClosestCenters = 0.f;
  this->dtClusterCenters = 0.f;
  this->dtCompactness = 0.f;
  this->dtTranspose = 0.f;
  this->dtReset = 0.f;
  this->start = NULL;
  this->stop = NULL;

  /* set the pointers to NULL */
  dev_data = NULL;
  dev_mmMult = NULL;
  dev_data_norm_squared = NULL;
  dev_partial_reduction = NULL;
  dev_partial_reduction_indices = NULL;
  dev_ccindex = NULL;
  dev_centers = NULL;
  dev_centers_transpose = NULL;
  dev_centers_large = NULL;
  dev_centers_norm_squared = NULL;
  dev_counts = NULL;
  dev_counts_large = NULL;
  dev_compactness = NULL;

  int dev;
  cudaGetDevice(&dev);
  cudaDeviceReset();
}

template<class TYPE>
kmeans<TYPE>::~kmeans() {
  
  /* Free the device data */
  if (this->dev_data) {
    if (cudaSuccess != cudaFree(this->dev_data))
      printf("%s(%i) : CUDA Runtime API error : %s.\n",
	     __FILE__, __LINE__, cudaGetErrorString(cudaGetLastError()));
  }
  
  if (this->dev_data_norm_squared) {
    if (cudaSuccess != cudaFree(this->dev_data_norm_squared))
      printf("%s(%i) : CUDA Runtime API error : %s.\n",
	     __FILE__, __LINE__, cudaGetErrorString(cudaGetLastError()));
  }

  if (this->dev_ccindex) {
    if (cudaSuccess != cudaFree(this->dev_ccindex))
      printf("%s(%i) : CUDA Runtime API error : %s.\n",
	     __FILE__, __LINE__, cudaGetErrorString(cudaGetLastError()));
  }



  /* centers data structures */
  if (this->dev_centers) {
    if (cudaSuccess != cudaFree(this->dev_centers))
      printf("%s(%i) : CUDA Runtime API error : %s.\n",
	     __FILE__, __LINE__, cudaGetErrorString(cudaGetLastError()));
  }

  if (this->dev_centers_transpose) {
    if (cudaSuccess != cudaFree(this->dev_centers_transpose))
      printf("%s(%i) : CUDA Runtime API error : %s.\n",
	     __FILE__, __LINE__, cudaGetErrorString(cudaGetLastError()));
  }

  if (this->dev_centers_large) {
    if (cudaSuccess != cudaFree(this->dev_centers_large))
      printf("%s(%i) : CUDA Runtime API error : %s.\n",
	     __FILE__, __LINE__, cudaGetErrorString(cudaGetLastError()));
  }

  if (this->dev_centers_norm_squared) {
    if (cudaSuccess != cudaFree(this->dev_centers_norm_squared))
      printf("%s(%i) : CUDA Runtime API error : %s.\n",
	     __FILE__, __LINE__, cudaGetErrorString(cudaGetLastError()));
  }

  /* counts data structure */
  if (this->dev_counts) {
    if (cudaSuccess != cudaFree(this->dev_counts))
      printf("%s(%i) : CUDA Runtime API error : %s.\n",
	     __FILE__, __LINE__, cudaGetErrorString(cudaGetLastError()));
  }

  if (this->dev_counts_large) {
    if (cudaSuccess != cudaFree(this->dev_counts_large))
      printf("%s(%i) : CUDA Runtime API error : %s.\n",
	     __FILE__, __LINE__, cudaGetErrorString(cudaGetLastError()));
  }

  /* compactness */
  if (this->dev_compactness) {
    if (cudaSuccess != cudaFree(this->dev_compactness))
      printf("%s(%i) : CUDA Runtime API error : %s.\n",
	     __FILE__, __LINE__, cudaGetErrorString(cudaGetLastError()));
  }

  /* Partial Reduction data structures */
  if (this->dev_partial_reduction) {
    if (cudaSuccess != cudaFree(this->dev_partial_reduction))
      printf("%s(%i) : CUDA Runtime API error : %s.\n",
	     __FILE__, __LINE__, cudaGetErrorString(cudaGetLastError()));
  }

  if (this->dev_partial_reduction_indices) {
    if (cudaSuccess != cudaFree(this->dev_partial_reduction_indices))
      printf("%s(%i) : CUDA Runtime API error : %s.\n",
	     __FILE__, __LINE__, cudaGetErrorString(cudaGetLastError()));
  }


  /* Events */
  if (this->start) {
    if (cudaSuccess != cudaEventDestroy(this->start))
      printf("%s(%i) : CUDA Runtime API error : %s.\n",
	     __FILE__, __LINE__, cudaGetErrorString(cudaGetLastError()));
  }

  if (this->stop) {
    if (cudaSuccess != cudaEventDestroy(this->stop))
      printf("%s(%i) : CUDA Runtime API error : %s.\n",
	     __FILE__, __LINE__, cudaGetErrorString(cudaGetLastError()));
  }

  cout << "Timings\n\tClosestCenters dt = " << this->dtClosestCenters
       << "\n\tClusterCenters dt = " << this->dtClusterCenters
       << "\n\tCompactness dt = " << this->dtCompactness
       << "\n\tTranspose dt = " << this->dtTranspose
       << "\n\tReset dt = " << this->dtReset << endl;


  /* create a cublas context */
  if (this->handle) {
    cublasStatus_t stat = cublasDestroy(this->handle);
    if (CUBLAS_STATUS_SUCCESS != stat)
      printf("%s(%i) : CUBLAS Runtime API error : %d.\n", __FILE__, __LINE__, int(stat));
  }

  if (this->useCUBLAS) {
    /* allocate the matrix matrix multiply result */
    if (this->dev_mmMult) {
      if (cudaSuccess != cudaFree(this->dev_mmMult))
	printf("%s(%i) : CUDA Runtime API error : %s.\n",
	       __FILE__, __LINE__, cudaGetErrorString(cudaGetLastError()));
    }
  }
}

/* copy the data back to the host */
template<class TYPE>
kmeansErrorStatus kmeans<TYPE>::getClusterCenters(TYPE * result)  {
  try {
    int nBytes = this->n * this->k * sizeof(TYPE);
    CUDA_API_SAFE_CALL(cudaMemcpy(result, this->dev_centers, nBytes, cudaMemcpyDeviceToHost), KMEANS_ERROR_MEMCPY);
    return KMEANS_SUCCESS;
  }
  catch (...) {
    return KMEANS_ERROR_MEMCPY;
  }
}


/* build the device data */
template<class TYPE> 
kmeansErrorStatus kmeans<TYPE>::computeTiling() {

  try {
    uint64_t fixedMemory = 0;
    fixedMemory += (uint64_t) (this->m * sizeof(TYPE));
    fixedMemory += (uint64_t) (this->m * sizeof(int));
    fixedMemory += (uint64_t) (this->n * this->k * sizeof(TYPE));
    fixedMemory += (uint64_t) (this->n * this->k * sizeof(TYPE));
    fixedMemory += (uint64_t) (getMaxConcurrentBlocks() * this->n * this->k * sizeof(TYPE));
    fixedMemory += (uint64_t) (this->k * sizeof(TYPE));
    fixedMemory += (uint64_t) (this->k * sizeof(int));
    fixedMemory += (uint64_t) (getMaxConcurrentBlocks() * this->k * sizeof(int));
    fixedMemory += (uint64_t) (getMaxConcurrentBlocks() * sizeof(TYPE));
    
    uint64_t varMemoryArray = 0;
    uint64_t varMemoryMMResult = 0;
    if (useCUBLAS) {
      varMemoryArray += (uint64_t) (this->m * this->n * sizeof(TYPE));
      varMemoryMMResult += (uint64_t) (this->m_padded * this->k * sizeof(TYPE));
    } else {
      varMemoryArray += (uint64_t) (this->m_padded * this->n * sizeof(TYPE));
      varMemoryMMResult += (uint64_t) (this->m * this->p * sizeof(TYPE));
      varMemoryMMResult += (uint64_t) (this->m * this->p * sizeof(int));
    }
    cout << "Fixed Memory = " << 1.0*fixedMemory/(1024*1024*1024) << " GBs" << endl;
    cout << "Array Memory = " << 1.0*varMemoryArray/(1024*1024*1024) << " GBs" << endl;
    cout << "Matrix Multiply Memory = " << 1.0*varMemoryMMResult/(1024*1024*1024) << " GBs" << endl;
    
    /* get the device memory */
    size_t freeMemory = 0;
    size_t totalMemory = 0;
    CUDA_API_SAFE_CALL(cudaMemGetInfo(&freeMemory, &totalMemory),KMEANS_ERROR_TILING);
    cout << "Free Memory = " << 1.0*freeMemory/(1024*1024*1024) << " GBs" << endl;
    cout << "Total Memory = " << 1.0*totalMemory/(1024*1024*1024) << " GBs" << endl;
    
    /* subdivide until all the arrays fit */
    /* the boundaries of the tiles must be a multiple of factor */
    uint64_t requiredMemory = fixedMemory + varMemoryArray + varMemoryMMResult;
    
  /* we will attempt to use at most 95% of the free memory */
    this->nTiles = 1;
    this->mChunk = (this->m + this->nTiles - 1)/this->nTiles;
    
    while (requiredMemory>.95*freeMemory) {
      /* increment the number of tiles */
      this->nTiles++;
      
      /* calculate the chunk size */
      this->mChunk = (this->m + this->nTiles - 1)/this->nTiles;
      this->m_padded = ((this->mChunk + this->factor - 1) / this->factor)*this->factor;
      
    /* update the memory requirements */
      if (useCUBLAS) {
	varMemoryArray = (uint64_t) (this->mChunk * this->n * sizeof(TYPE));
	varMemoryMMResult = (uint64_t) (this->mChunk * this->k * sizeof(TYPE));
      } else {
	varMemoryArray = (uint64_t) (this->m_padded * this->n * sizeof(TYPE));
	varMemoryMMResult = (uint64_t) (this->m_padded * this->p * sizeof(TYPE));
	varMemoryMMResult += (uint64_t) (this->m_padded * this->p * sizeof(int));
      }
      
      /* calculate the new required memory */
      requiredMemory = fixedMemory + varMemoryArray + varMemoryMMResult;
    }
    return KMEANS_SUCCESS;
  }
  catch (...) {
    return KMEANS_ERROR_TILING;
  }
}

/* build the device data */
template<class TYPE>
kmeansErrorStatus kmeans<TYPE>::buildData(const TYPE * data) {

  try {
    /*******************/
    /* allocate timers */
    /*******************/
    CUDA_API_SAFE_CALL(cudaEventCreate(&this->start), KMEANS_ERROR_BUILDDATA);
    CUDA_API_SAFE_CALL(cudaEventCreate(&this->stop), KMEANS_ERROR_BUILDDATA);

    /*******************/
    /* allocate arrays */
    /*******************/

    /* norm squared of each row of the input data */
    uint64_t nBytes = this->m * sizeof(TYPE);
    CUDA_API_SAFE_CALL(cudaMalloc((void**)&(this->dev_data_norm_squared), nBytes), KMEANS_ERROR_BUILDDATA);
    CUDA_API_SAFE_CALL(cudaMemset(this->dev_data_norm_squared, 0, nBytes), KMEANS_ERROR_BUILDDATA);

    /* the index of the closest cluster center */
    nBytes = this->m * sizeof(int);
    CUDA_API_SAFE_CALL(cudaMalloc((void**)&(this->dev_ccindex), nBytes), KMEANS_ERROR_BUILDDATA);
    CUDA_API_SAFE_CALL(cudaMemset(this->dev_ccindex, 0, nBytes), KMEANS_ERROR_BUILDDATA);

    /* the device cluster centers */
    nBytes = this->n * this->k * sizeof(TYPE);
    CUDA_API_SAFE_CALL(cudaMalloc((void**)&(this->dev_centers), nBytes), KMEANS_ERROR_BUILDDATA);
    CUDA_API_SAFE_CALL(cudaMemset(this->dev_centers, 0, nBytes), KMEANS_ERROR_BUILDDATA);

    /* the device cluster centers transposed */
    CUDA_API_SAFE_CALL(cudaMalloc((void**)&(this->dev_centers_transpose), nBytes), KMEANS_ERROR_BUILDDATA);
    CUDA_API_SAFE_CALL(cudaMemset(this->dev_centers_transpose, 0, nBytes), KMEANS_ERROR_BUILDDATA);

    /* the large device cluster centers */
    nBytes = getMaxConcurrentBlocks() * this->n * this->k * sizeof(TYPE);
    CUDA_API_SAFE_CALL(cudaMalloc((void**)&(this->dev_centers_large), nBytes), KMEANS_ERROR_BUILDDATA);
    CUDA_API_SAFE_CALL(cudaMemset(this->dev_centers_large, 0, nBytes), KMEANS_ERROR_BUILDDATA);

    /* norm squared of each cluster center */
    nBytes = this->k * sizeof(TYPE);
    CUDA_API_SAFE_CALL(cudaMalloc((void**)&(this->dev_centers_norm_squared), nBytes), KMEANS_ERROR_BUILDDATA);
    CUDA_API_SAFE_CALL(cudaMemset(this->dev_centers_norm_squared, 0, nBytes), KMEANS_ERROR_BUILDDATA);

    /* the counts of the data which contribute to each cluster center */
    nBytes = this->k * sizeof(int);
    CUDA_API_SAFE_CALL(cudaMalloc((void**)&(this->dev_counts), nBytes), KMEANS_ERROR_BUILDDATA);
    CUDA_API_SAFE_CALL(cudaMemset(this->dev_counts, 0, nBytes), KMEANS_ERROR_BUILDDATA);

    /* the large counts of the data which contribute to each cluster center */
    nBytes = getMaxConcurrentBlocks() * this->k * sizeof(int);
    CUDA_API_SAFE_CALL(cudaMalloc((void**)&(this->dev_counts_large), nBytes), KMEANS_ERROR_BUILDDATA);
    CUDA_API_SAFE_CALL(cudaMemset(this->dev_counts_large, 0, nBytes), KMEANS_ERROR_BUILDDATA);

    /* the compactness calculation array */
    nBytes = getMaxConcurrentBlocks() * sizeof(TYPE);
    CUDA_API_SAFE_CALL(cudaMalloc((void**)&(this->dev_compactness), nBytes), KMEANS_ERROR_BUILDDATA);
    CUDA_API_SAFE_CALL(cudaMemset(this->dev_compactness, 0, nBytes), KMEANS_ERROR_BUILDDATA);

    /*********************************/
    /* create CUBLAS library context */
    /*********************************/
    CUBLAS_API_SAFE_CALL(cublasCreate(&this->handle), KMEANS_ERROR_BUILDDATA);

    /* allocate space for the result of the matrix matrix multiply */
    if (this->useCUBLAS) {
      /* input data */
      nBytes = this->m * this->n * sizeof(TYPE);
      CUDA_API_SAFE_CALL(cudaMalloc((void**)&(this->dev_data), nBytes), KMEANS_ERROR_BUILDDATA);
      CUDA_API_SAFE_CALL(cudaMemset(this->dev_data, 0, nBytes), KMEANS_ERROR_BUILDDATA);
      
      nBytes = this->m * this->n * sizeof(TYPE);
      CUDA_API_SAFE_CALL(cudaMemcpy(this->dev_data, data, nBytes, cudaMemcpyHostToDevice), KMEANS_ERROR_BUILDDATA);

      /* matrix matrix multiply buffer */
      nBytes = this->m * this->k * sizeof(TYPE);
      CUDA_API_SAFE_CALL(cudaMalloc((void**)&(this->dev_mmMult), nBytes), KMEANS_ERROR_BUILDDATA);
      CUDA_API_SAFE_CALL(cudaMemset(dev_mmMult, 0, nBytes), KMEANS_ERROR_BUILDDATA);
    } else {
      /* input data */
      nBytes = this->m_padded * this->n * sizeof(TYPE);
      CUDA_API_SAFE_CALL(cudaMalloc((void**)&(this->dev_data), nBytes), KMEANS_ERROR_BUILDDATA);
      CUDA_API_SAFE_CALL(cudaMemset(this->dev_data, 0, nBytes), KMEANS_ERROR_BUILDDATA);
      
      nBytes = this->m * this->n * sizeof(TYPE);
      CUDA_API_SAFE_CALL(cudaMemcpy(this->dev_data, data, nBytes, cudaMemcpyHostToDevice), KMEANS_ERROR_BUILDDATA);

      /* the partial reduction array */
      nBytes = this->m * this->p * sizeof(TYPE);
      CUDA_API_SAFE_CALL(cudaMalloc((void**)&(this->dev_partial_reduction), nBytes), KMEANS_ERROR_BUILDDATA);
      CUDA_API_SAFE_CALL(cudaMemset(this->dev_partial_reduction, 0, nBytes), KMEANS_ERROR_BUILDDATA);
      
      /* the partial reduction array indices */
      nBytes = this->m * this->p * sizeof(int);
      CUDA_API_SAFE_CALL(cudaMalloc((void**)&(this->dev_partial_reduction_indices), nBytes), KMEANS_ERROR_BUILDDATA);
      CUDA_API_SAFE_CALL(cudaMemset(this->dev_partial_reduction_indices, 0, nBytes), KMEANS_ERROR_BUILDDATA);
    }

    return KMEANS_SUCCESS;
  }
  catch (...) {
    return KMEANS_ERROR_BUILDDATA;
  }
}

/* initialize */
template<class TYPE>
kmeansErrorStatus kmeans<TYPE>::initialize(const TYPE * data, TYPE * result) {
  try {
    kmeansCudaErrorStatus err = NO_ERROR;

    /* normalize the input data */
    err = rowNormalize<TYPE>(this->m, this->n, this->dev_data,
			     this->dev_data_norm_squared);
    if (err != NO_ERROR) return KMEANS_ERROR_INITIALIZE;

    /* initialize the cluster centers */
    vector<int> points(0);
    srand(42);
    for (int i = 0; i < this->k; ++i) {
      int index = (int)floor(rand() % this->m);
      while (binary_search(points.begin(), points.end(), index))
	index = (int)floor(rand() % this->m);
      points.push_back(index);
    }

    /* copy the data and normalize */
    for (int i = 0; i < this->k; ++i)
      for (int j = 0; j < this->n; ++j)
	result[j*this->k + i] = data[points[i] * this->n + j];

    /* copy the data to the GPU */
    int nBytes = this->k * this->n * sizeof(TYPE);
    CUDA_API_SAFE_CALL(cudaMemcpy(this->dev_centers, result, nBytes, cudaMemcpyHostToDevice),
		       KMEANS_ERROR_INITIALIZE);

    /* normalize the centers data */
    err = colNormalize<TYPE>(this->n, this->k, this->dev_centers,
			     this->dev_centers_norm_squared);
    if (err != NO_ERROR) return KMEANS_ERROR_INITIALIZE;

#if 0
    /* debug */
    vector<TYPE> x(this->m);
    vector<TYPE> y(this->k);

    CUDA_API_SAFE_CALL(cudaMemcpy(x.data(),this->dev_data_norm_squared,
				  this->m*sizeof(TYPE),cudaMemcpyDeviceToHost),
		       KMEANS_ERROR_INITIALIZE);

    CUDA_API_SAFE_CALL(cudaMemcpy(y.data(),this->dev_centers_norm_squared,
				  this->k*sizeof(TYPE),cudaMemcpyDeviceToHost),
		       KMEANS_ERROR_INITIALIZE);

    for (int i=0; i<this->k; ++i) {
      if (fabs(y[i]-x[points[i]])/fabs(y[i]) > 1.e-6)
	cout << i << " " << points[i] << " : " << x[points[i]] << " " << y[i] << endl;
    }
#endif
    return KMEANS_SUCCESS;
  }
  catch (...) {
    return KMEANS_ERROR_INITIALIZE;
  }
}

/* compute */
template<class TYPE>
kmeansErrorStatus kmeans<TYPE>::closestCenters() {

  kmeansCudaErrorStatus err = NO_ERROR;

  /* start the timer */
  float DT = 0.f;
  CUDA_API_SAFE_CALL(cudaEventRecord(this->start, 0), KMEANS_ERROR_CLOSEST_CENTERS);

  if (this->useCUBLAS) {
    TYPE one = 1.0;
    TYPE zero = 0.0;
    if (sizeof(TYPE) == 4) {
      CUBLAS_API_SAFE_CALL(cublasSgemm(this->handle, CUBLAS_OP_N, CUBLAS_OP_N,
				       this->k, this->m, this->n, (const float *)&one,
				       (const float *) this->dev_centers, this->k,
				       (const float *) this->dev_data, this->n,
				       (const float *)&zero, (float *) this->dev_mmMult, this->k),
			   KMEANS_ERROR_CLOSEST_CENTERS);

      err = rowTransformMinimumF(this->m, this->k,
				 (const float *)this->dev_data_norm_squared,
				 (const float *)this->dev_centers_norm_squared,
				 (const float *)this->dev_mmMult,
				 (int *) this->dev_ccindex);

    }
    else {
      CUBLAS_API_SAFE_CALL(cublasDgemm(this->handle, CUBLAS_OP_N, CUBLAS_OP_N,
				       this->k, this->m, this->n, (const double *)&one,
				       (const double *) this->dev_centers, this->k,
				       (const double *) this->dev_data, this->n,
				       (const double *)&zero, (double *) this->dev_mmMult, this->k),
			   KMEANS_ERROR_CLOSEST_CENTERS);

      err = rowTransformMinimumD(this->m, this->k,
				 (const double *)this->dev_data_norm_squared,
				 (const double *)this->dev_centers_norm_squared,
				 (const double *)this->dev_mmMult,
				 (int *) this->dev_ccindex);
    }
  }
  else {
    err = ClosestCenters<TYPE>(this->m, this->n, this->dev_data,
			       this->k, this->dev_centers,
			       this->dev_data_norm_squared,
			       this->dev_centers_norm_squared,
			       this->p, this->dev_partial_reduction,
			       this->dev_partial_reduction_indices,
			       this->dev_ccindex, this->constantMemSet);
  }

  /* stop the timer */
  CUDA_API_SAFE_CALL(cudaEventRecord(this->stop, 0), KMEANS_ERROR_CLOSEST_CENTERS);
  CUDA_API_SAFE_CALL(cudaEventSynchronize(this->stop), KMEANS_ERROR_CLOSEST_CENTERS);
  CUDA_API_SAFE_CALL(cudaEventElapsedTime(&DT, this->start, this->stop), KMEANS_ERROR_CLOSEST_CENTERS);
  this->dtClosestCenters += ((float).001)*DT;

  if (err != NO_ERROR) return KMEANS_ERROR_CLOSEST_CENTERS;
  return KMEANS_SUCCESS;
}


/* cluster centers */
template<class TYPE>
kmeansErrorStatus kmeans<TYPE>::clusterCenters() {

  kmeansCudaErrorStatus err = NO_ERROR;

  /* start the timer */
  float DT = 0.f;
  CUDA_API_SAFE_CALL(cudaEventRecord(this->start, 0), KMEANS_ERROR_CLUSTER_CENTERS);

  /* compute the new cluster centers */
  err = ClusterCenters<TYPE>(this->m, this->n, this->k,
			     this->dev_data, this->dev_ccindex,
			     this->dev_centers_large, this->dev_counts_large,
			     this->dev_centers_transpose, this->dev_counts);

  /* stop the timer */
  CUDA_API_SAFE_CALL(cudaEventRecord(this->stop, 0), KMEANS_ERROR_CLUSTER_CENTERS);
  CUDA_API_SAFE_CALL(cudaEventSynchronize(this->stop), KMEANS_ERROR_CLUSTER_CENTERS);
  CUDA_API_SAFE_CALL(cudaEventElapsedTime(&DT, this->start, this->stop), KMEANS_ERROR_CLUSTER_CENTERS);
  this->dtClusterCenters += ((float).001)*DT;

  if (err != NO_ERROR) return KMEANS_ERROR_CLUSTER_CENTERS;
  return KMEANS_SUCCESS;
}

/* compactness */
template<class TYPE>
kmeansErrorStatus kmeans<TYPE>::compactness() {

  kmeansCudaErrorStatus err = NO_ERROR;

  /* start the timer */
  float DT = 0.f;
  CUDA_API_SAFE_CALL(cudaEventRecord(this->start, 0), KMEANS_ERROR_COMPACTNESS);

  /* compute the compactness */
  err = Compactness<TYPE>(this->m, this->n, this->k,
			  this->dev_data, this->dev_ccindex,
			  this->dev_centers_transpose, this->dev_compactness,
			  this->host_compactness.data());

  /* stop the timer */
  CUDA_API_SAFE_CALL(cudaEventRecord(this->stop, 0), KMEANS_ERROR_COMPACTNESS);
  CUDA_API_SAFE_CALL(cudaEventSynchronize(this->stop), KMEANS_ERROR_COMPACTNESS);
  CUDA_API_SAFE_CALL(cudaEventElapsedTime(&DT, this->start, this->stop), KMEANS_ERROR_COMPACTNESS);
  this->dtCompactness += ((float).001)*DT;

  if (err != NO_ERROR) return KMEANS_ERROR_COMPACTNESS;
  return KMEANS_SUCCESS;
}


/* compactness */
template<class TYPE>
kmeansErrorStatus kmeans<TYPE>::transpose() {

  kmeansCudaErrorStatus err = NO_ERROR;

  /* start the timer */
  float DT = 0.f;
  CUDA_API_SAFE_CALL(cudaEventRecord(this->start, 0), KMEANS_ERROR_TRANSPOSE);
  TYPE one = 1.0;
  TYPE zero = 0.0;

  /* compute the transpose */
  if (sizeof(TYPE) == 4) {
    CUBLAS_API_SAFE_CALL(cublasSgeam(this->handle, CUBLAS_OP_T, CUBLAS_OP_T, this->k, this->n,
				     (const float *)&one, (const float *) this->dev_centers_transpose, this->n,
				     (const float *)&zero, (const float *) this->dev_centers_transpose, this->n,
				     (float *) this->dev_centers, this->k),
			 KMEANS_ERROR_TRANSPOSE);
  }
  else {
    CUBLAS_API_SAFE_CALL(cublasDgeam(this->handle, CUBLAS_OP_T, CUBLAS_OP_T, this->k, this->n,
				     (const double *)&one, (const double *) this->dev_centers_transpose, this->n,
				     (const double *)&zero, (const double *) this->dev_centers_transpose, this->n,
				     (double *) this->dev_centers, this->k),
			 KMEANS_ERROR_TRANSPOSE);
  }
  /* stop the timer */
  CUDA_API_SAFE_CALL(cudaEventRecord(this->stop, 0), KMEANS_ERROR_TRANSPOSE);
  CUDA_API_SAFE_CALL(cudaEventSynchronize(this->stop), KMEANS_ERROR_TRANSPOSE);
  CUDA_API_SAFE_CALL(cudaEventElapsedTime(&DT, this->start, this->stop), KMEANS_ERROR_TRANSPOSE);
  this->dtTranspose += ((float).001)*DT;

  if (err != NO_ERROR) return KMEANS_ERROR_TRANSPOSE;
  return KMEANS_SUCCESS;
}
/* reset */
template<class TYPE>
kmeansErrorStatus kmeans<TYPE>::reset() {

  int nBytes = 0;
  kmeansCudaErrorStatus err = NO_ERROR;

  /* start the timer */
  float DT = 0.f;
  CUDA_API_SAFE_CALL(cudaEventRecord(this->start, 0), KMEANS_ERROR_RESET);

  /* Reset this data to 0 at every iteration */
  nBytes = getMaxConcurrentBlocks() * this->n * this->k * sizeof(TYPE);
  CUDA_API_SAFE_CALL(cudaMemset(this->dev_centers_large, 0, nBytes), KMEANS_ERROR_RESET);

  nBytes = getMaxConcurrentBlocks() * this->k * sizeof(int);
  CUDA_API_SAFE_CALL(cudaMemset(this->dev_counts_large, 0, nBytes), KMEANS_ERROR_RESET);

  nBytes = getMaxConcurrentBlocks() * sizeof(TYPE);
  CUDA_API_SAFE_CALL(cudaMemset(this->dev_compactness, 0, nBytes), KMEANS_ERROR_RESET);

  /* normalize the cluster centers */
  err = colNormalize<TYPE>(this->n, this->k, this->dev_centers,
			   this->dev_centers_norm_squared);

  /* stop the timer */
  CUDA_API_SAFE_CALL(cudaEventRecord(this->stop, 0), KMEANS_ERROR_RESET);
  CUDA_API_SAFE_CALL(cudaEventSynchronize(this->stop), KMEANS_ERROR_RESET);
  CUDA_API_SAFE_CALL(cudaEventElapsedTime(&DT, this->start, this->stop), KMEANS_ERROR_RESET);
  this->dtReset += ((float).001)*DT;

  if (err != NO_ERROR) return KMEANS_ERROR_RESET;
  return KMEANS_SUCCESS;
}

/* compute */
template<class TYPE>
kmeansErrorStatus kmeans<TYPE>::compute() {

  try {
    kmeansErrorStatus err = KMEANS_SUCCESS;
    int iters = 0;
    while (iters < this->maxIters && this->relErr>this->criterion) {

      /* compute the closest center */
      err = reset();
      if (err != KMEANS_SUCCESS) return err;

      /* compute the closest center */
      err = closestCenters();
      if (err != KMEANS_SUCCESS) return err;

      /* compute the closest center */
      err = clusterCenters();
      if (err != KMEANS_SUCCESS) return err;

      /* compute the compactness */
      err = compactness();
      if (err != KMEANS_SUCCESS) return err;

      /* transpose the results */
      err = transpose();
      if (err != KMEANS_SUCCESS) return err;

      /* compute the convergence statistic */
      TYPE compactness_old = this->compactnessScore;
      this->compactnessScore = this->host_compactness[0];
      this->relErr = fabs(this->compactnessScore - compactness_old) / fabs(this->compactnessScore);
      iters++;
    }
    cout << "iteration " << iters << " relErr = " << this->relErr
	 << " compactness = " << this->compactnessScore << endl;
    return KMEANS_SUCCESS;
  }
  catch (...) {
    return KMEANS_ERROR_COMPUTE;
  }
}


DllExport kmeansErrorStatus KMEANS_API computeKmeansF(const float * data, const int m, const int n,
						      const int k, const float criterion,
						      const int maxIters, const int numRetries,
						      const int initAlgorithm, const int useCUBLAS,
						      float * result) {

  kmeansErrorStatus err = KMEANS_SUCCESS;

  /* Allocate an instance of kmeans */
  kmeans<float> * KMEANS = new kmeans<float>(m, n, k, criterion, maxIters,
					     numRetries, initAlgorithm, useCUBLAS);

  /* compute the tiling */
  err = KMEANS->computeTiling();
  if (err != KMEANS_SUCCESS) goto cleanup;

  /* build the device data */
  err = KMEANS->buildData(data);
  if (err != KMEANS_SUCCESS) goto cleanup;

  /* initialize */
  err = KMEANS->initialize(data, result);
  if (err != KMEANS_SUCCESS) goto cleanup;

  /* compute */
  err = KMEANS->compute();
  if (err != KMEANS_SUCCESS) goto cleanup;

  /* transfer the cluster centers data back to the host */
  err = KMEANS->getClusterCenters(result);
  if (err != KMEANS_SUCCESS) goto cleanup;

  /* determine whether the iteration converged */
  err = KMEANS->getConvergenceFlag();
  if (err != KMEANS_SUCCESS) goto cleanup;

  goto cleanup;

 cleanup:

  /* delete the instance of kmeans */
  delete KMEANS;
  return err;
}


DllExport kmeansErrorStatus KMEANS_API computeKmeansD(const double * data, const int m, const int n,
						      const int k, const double criterion,
						      const int maxIters, const int numRetries,
						      const int initAlgorithm, const int useCUBLAS,
						      double * result) {

  kmeansErrorStatus err = KMEANS_SUCCESS;

  /* Allocate an instance of kmeans */
  kmeans<double> * KMEANS = new kmeans<double>(m, n, k, criterion, maxIters,
					       numRetries, initAlgorithm, useCUBLAS);

  /* compute the tiling */
  err = KMEANS->computeTiling();
  if (err != KMEANS_SUCCESS) goto cleanup;

  /* build the device data */
  err = KMEANS->buildData(data);
  if (err != KMEANS_SUCCESS) goto cleanup;

  /* initialize */
  err = KMEANS->initialize(data, result);
  if (err != KMEANS_SUCCESS) goto cleanup;

  /* compute */
  err = KMEANS->compute();
  if (err != KMEANS_SUCCESS) goto cleanup;

  /* transfer the cluster centers data back to the host */
  err = KMEANS->getClusterCenters(result);
  if (err != KMEANS_SUCCESS) goto cleanup;

  /* determine whether the iteration converged */
  err = KMEANS->getConvergenceFlag();
  if (err != KMEANS_SUCCESS) goto cleanup;

  goto cleanup;

 cleanup:

  /* delete the instance of kmeans */
  delete KMEANS;
  return err;
}

