#include "kmeans.hpp"

template<class TYPE>
kmeans<TYPE>::kmeans(const int m, const int n, const int k,
		     const TYPE criterion, const int maxIters,
		     const int numRetries, const int initAlgorithm,
		     const int useCUBLAS, const TYPE miniBatchFraction) {

  /* reset the device */
  //int dev;
  //cudaGetDevice(&dev);
  //cudaDeviceReset();

  /* allocate timers */
  this->start = NULL;
  this->stop = NULL;
  cudaError_t err = cudaEventCreate(&this->start);
  if (err!=cudaSuccess) cout << "Error in cudaEventCreate in constructor." << endl;
  err = cudaEventCreate(&this->stop);
  if (err!=cudaSuccess) cout << "Error in cudaEventCreate in constructor." << endl;

  /* start the timer */
  float DT = 0.f;
  cudaEventRecord(this->start, 0);


  /* set the key meta data from the input */
  this->m = m;
  this->n = n;
  this->k = k;
  
  this->criterion = criterion;
  this->maxIters = maxIters;
  this->numRetries = numRetries;

  if (sizeof(TYPE) == 4)
    this->factor = TILESIZE*N_UNROLL_FLOAT;
  else
    this->factor = TILESIZE*N_UNROLL_DOUBLE;

  this->p = (this->k + this->factor - 1) / this->factor;
  this->m_padded = ((this->m + this->factor - 1) / this->factor)*this->factor;
  
  /* set the convergence meta data */
  this->compactnessScore = 0.;
  this->relErr = 1.0;
  

  this->handle = NULL;
  cout << "useCUBLAS=" << useCUBLAS << endl;
  if (useCUBLAS) this->useCUBLAS = true;
  else this->useCUBLAS = false;
  
  this->constantMemSet = false;

  this->entireDatasetIsOnDevice = false;

  this->smallMiniBatch = false;

  /* mini batch parameters */
  if (miniBatchFraction<1.0) {
    this->useMiniBatch = true;
    this->miniBatchFraction = miniBatchFraction;
    this->m_mb = (int)(this->miniBatchFraction*this->m);
    this->m_mb_padded = ((this->m_mb + this->factor - 1) / this->factor)*this->factor;
    this->m_mb_index = 0;


    std::cout << "miniBatchFraction = " << this->miniBatchFraction << std::endl;
    std::cout << "\tm =" << this->m << std::endl;
    std::cout << "\tm_mb =" << this->m_mb << std::endl;
    std::cout << "\tm_mb_padded =" << this->m_mb_padded << std::endl;

  } else {
    this->useMiniBatch = false;
    this->miniBatchFraction = 1.0f;
    this->m_mb = 0;
    this->m_mb_padded = 0;
  }
    
  /* timing data */
  this->dtClosestCenters = 0.f;
  this->dtClusterCenters = 0.f;
  this->dtCompactness = 0.f;
  this->dtTranspose = 0.f;
  this->dtReset = 0.f;
  this->dtCopy = 0.f;
  this->dtConstructMiniBatch = 0.f;
  this->dtBuild = 0.f;
  this->dtRowNormalize = 0.f;
  this->dtInitialize = 0.f;
  this->dtGetCenters = 0.f;

  /* set the pointers to NULL */
  this->host_data = NULL;
  this->dev_data = NULL;
  this->dev_data_mb = NULL;
  this->dev_mmMult = NULL;
  this->dev_data_norm_squared = NULL;
  this->dev_data_norm_squared_mb = NULL;
  this->dev_partial_reduction = NULL;
  this->dev_partial_reduction_indices = NULL;
  this->dev_ccindex = NULL;
  this->dev_centers = NULL;
  this->dev_centers_transpose = NULL;
  this->dev_centers_large = NULL;
  this->dev_centers_norm_squared = NULL;
  this->dev_counts = NULL;
  this->dev_counts_large = NULL;
  this->dev_compactness = NULL;

  /* stop the timer */
  cudaEventRecord(this->stop, 0);
  cudaEventSynchronize(this->stop);
  cudaEventElapsedTime(&DT, this->start, this->stop);
  this->dtBuild = ((float).001)*DT;
}

template<class TYPE>
kmeans<TYPE>::~kmeans() {

  
  /* Free the host data */
  if (this->host_data) delete [] this->host_data;

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

  float totalDt = this->dtReset + this->dtCopy + this->dtTranspose + this->dtCompactness + this->dtClusterCenters + this->dtClosestCenters  + this->dtBuild + this->dtRowNormalize + this->dtInitialize + this->dtConstructMiniBatch + this->dtGetCenters;

  cout << "Timings\n\tBuild dt = " << this->dtBuild
       << "\n\tRowNormalize dt = " << this->dtRowNormalize
       << "\n\tInitialize dt = " << this->dtInitialize
       << "\n\tClosestCenters dt = " << this->dtClosestCenters
       << "\n\tClusterCenters dt = " << this->dtClusterCenters
       << "\n\tCompactness dt = " << this->dtCompactness
       << "\n\tTranspose dt = " << this->dtTranspose
       << "\n\tData Xfer dt = " << this->dtCopy
       << "\n\tConstructMiniBatch dt = " << this->dtConstructMiniBatch
       << "\n\tReset dt = " << this->dtReset 
       << "\n\tGetCenters dt = " << this->dtGetCenters
       << "\n\tTotal dt = " << totalDt
       << endl;
}

/* copy the data back to the host */
template<class TYPE>
kmeansErrorStatus kmeans<TYPE>::getClusterCenters(TYPE * result)  {
  try {
    /* start the timer */
    START_TIMER(this->DT,this->start,KMEANS_ERROR_GET_CENTERS);

    int nBytes = this->n * this->k * sizeof(TYPE);
    CUDA_API_SAFE_CALL(cudaMemcpy(result, this->dev_centers, nBytes, cudaMemcpyDeviceToHost), KMEANS_ERROR_GET_CENTERS);

    /* stop the timer */
    STOP_TIMER(this->DT,this->start,this->stop,this->dtGetCenters,KMEANS_ERROR_GET_CENTERS);
    return KMEANS_SUCCESS;
  }
  catch (...) {
    return KMEANS_ERROR_GET_CENTERS;
  }
}


/* build the device data */
template<class TYPE> 
kmeansErrorStatus kmeans<TYPE>::computeTiling() {

  try {
    /* start the timer */
    START_TIMER(this->DT,this->start,KMEANS_ERROR_TILING);

    uint64_t fixedMemory = 0;
    int mCompute = this->useMiniBatch ? this->m_mb : this->m;
    int mComputePadded = this->useMiniBatch ? this->m_mb_padded : this->m_padded;

    fixedMemory += (uint64_t) (2*this->m * sizeof(TYPE));
    fixedMemory += (uint64_t) (this->m * sizeof(int));
    fixedMemory += (uint64_t) (this->n * this->k * sizeof(TYPE));
    fixedMemory += (uint64_t) (this->n * this->k * sizeof(TYPE));
    fixedMemory += (uint64_t) (getMaxConcurrentBlocks() * this->n * this->k * sizeof(TYPE));
    fixedMemory += (uint64_t) (this->k * sizeof(TYPE));
    fixedMemory += (uint64_t) (this->k * sizeof(int));
    fixedMemory += (uint64_t) (getMaxConcurrentBlocks() * this->k * sizeof(int));
    fixedMemory += (uint64_t) (getMaxConcurrentBlocks() * sizeof(TYPE));
    
    /* get the device memory */
    size_t freeMemory = 0;
    size_t totalMemory = 0;
    CUDA_API_SAFE_CALL(cudaMemGetInfo(&freeMemory, &totalMemory),KMEANS_ERROR_TILING);

    /* subdivide until all the arrays fit */
    /* the boundaries of the tiles must be a multiple of factor */
    uint64_t requiredMemory = fixedMemory;

    if (fixedMemory>.95*freeMemory) {
      std::cout << "Required fixed memory exceeds device limits. If using minibatch, try lowering the minibatch fraction. Exiting." << std::endl;
      return KMEANS_ERROR_TILING;
    }

    /* variable memory */
    uint64_t varMemoryArray = 0;
    uint64_t varMemoryMMResult = 0;
    if (useCUBLAS) {
      varMemoryMMResult += (uint64_t) (mComputePadded * this->k * sizeof(TYPE));
    } else {
      varMemoryMMResult += (uint64_t) (mComputePadded * this->p * sizeof(TYPE));
      varMemoryMMResult += (uint64_t) (mComputePadded * this->p * sizeof(int));
    }

    if (this->useMiniBatch) {

      /********************/
      /*  Using Minibatch */
      /********************/

      int MM = this->m + this->m_mb;
      int MMpadded = ((MM + this->factor - 1) / this->factor)*this->factor;

      varMemoryArray = (uint64_t) (MMpadded * this->n * sizeof(TYPE));
      requiredMemory =  varMemoryArray + varMemoryMMResult + fixedMemory;

      if (requiredMemory <= .95*freeMemory) {
	this->num_mb_per_tile = 1;
	this->smallMiniBatch = true;

	/* determine the tile start positions and the number of rows to compute in a tile */
	this->nTiles = 1;
	
	this->mTile.resize(this->nTiles);
	fill(this->mTile.begin(), this->mTile.end(), this->m);
	
	this->mStart.resize(this->nTiles);
	fill(this->mStart.begin(), this->mStart.end(), 0); 
	
      } else {

	this->num_mb_per_tile = 0;
	uint64_t varMemoryArray2 = 0;
	uint64_t requiredMemory2 = 0;
	this->smallMiniBatch = false;
	
	do {
	  this->num_mb_per_tile++;
	  int MM       = this->num_mb_per_tile*mCompute;
	  int MMpadded = ((MM + this->factor - 1) / this->factor)*this->factor;
	  
	  int MM2       = (this->num_mb_per_tile+1)*mCompute;
	  int MMpadded2 = ((MM2 + this->factor - 1) / this->factor)*this->factor;
	  
	  varMemoryArray = (uint64_t) (MMpadded * this->n * sizeof(TYPE));      
	  varMemoryArray2 = (uint64_t) (MMpadded2 * this->n * sizeof(TYPE));      
	  
	  /* total required memory */
	  requiredMemory =  varMemoryArray + varMemoryMMResult + fixedMemory;
	  requiredMemory2 =  varMemoryArray2 + varMemoryMMResult + fixedMemory;
	  
	  if (this->num_mb_per_tile==1 && requiredMemory>.95*freeMemory) {
	    std::cout << "Required fixed memory exceeds device limits. If using minibatch, try lowering the minibatch fraction. Exiting." << std::endl;
	    return KMEANS_ERROR_TILING;
	  }
	  
	} while (requiredMemory<=.95*freeMemory && requiredMemory2<=.95*freeMemory &&
		 this->num_mb_per_tile*this->m_mb<this->m);

	/* determine the tile start positions and the number of rows to compute in a tile */
	int tileSize = this->num_mb_per_tile*this->m_mb;
	this->nTiles = ((this->m + tileSize-1)/tileSize);
	
	this->mTile.resize(this->nTiles);
	fill(this->mTile.begin(), this->mTile.end(), tileSize);
	this->mTile.back() = this->m - (this->nTiles-1)*tileSize;
	
	this->mStart.resize(this->nTiles);
	fill(this->mStart.begin(), this->mStart.end(), 0); 
	for (int i=1; i<this->nTiles; ++i)
	  this->mStart[i] = this->mStart[i-1] + this->mTile[i-1];
	
#if 0
	for (int i=0; i<this->nTiles; ++i) {
	  cout << i+1 << "/" << this->nTiles << " " << this->mStart[i] << " " 
	       << this->mTile[i] << " " << this->mStart[i] + this->mTile[i] << endl;
	}
#endif
      }


      /* minibatch start and stop points */
      int iters = 0;
      int mCurrent = 0;
      do {
	this->mMBStart.push_back(mCurrent);
	mCurrent += this->m_mb*this->num_mb_per_tile;
	if (mCurrent>this->m) {
	  this->mMBWrapTile.push_back(true);   
	  mCurrent = mCurrent%(this->m);
	  iters++;	
	} else if (mCurrent==this->m) {
	  this->mMBWrapTile.push_back(true);   
	  mCurrent = 0;
	  iters++;	
	} else {
	  this->mMBWrapTile.push_back(false);
	}
#if 0
	std::cout << "iters=" << iters << " : " << this->mMBStart.back() 
		  << " " << this->mMBWrapTile.back() << std::endl;
#endif
      } while (iters<=this->maxIters);

#if 0
      std::cout << "num MBs per tile = " << this->num_mb_per_tile << std::endl;
      std::cout << "freeMemory = " << freeMemory/(1024.*1024.*1024.) << std::endl;
      std::cout << "fixedMemory = " << fixedMemory/(1024.*1024.*1024.) << std::endl;
      std::cout << "varMemoryArray = " << varMemoryArray/(1024.*1024.*1024.) << std::endl;
      std::cout << "varMemoryMMResult = " << varMemoryMMResult/(1024.*1024.*1024.) << std::endl;
      std::cout << "requiredMemory = " << requiredMemory/(1024.*1024.*1024.) << std::endl;
#endif


    } else {

      /*********************************/
      /*  Normal (non mini batch) Code */
      /*********************************/

      /* variable memory */
      varMemoryArray += (uint64_t) (this->m_padded * this->n * sizeof(TYPE));      

      /* total required memory */
      requiredMemory +=  varMemoryArray + varMemoryMMResult;

      /* loop until required memroy is less than 95% of free memory */
      this->nTiles=1;
      while (requiredMemory>.95*freeMemory) {

	/* increment the number of tiles */
	this->nTiles++;
	
	/* calculate the chunk size */
	int mChunk = (this->m + this->nTiles - 1)/this->nTiles;
	this->m_padded = ((mChunk + this->factor - 1) / this->factor)*this->factor;
	
	/* update the memory requirements */
	varMemoryArray = (uint64_t) (this->m_padded * this->n * sizeof(TYPE));
	if (useCUBLAS) {
	  varMemoryMMResult = (uint64_t) (this->m_padded * this->k * sizeof(TYPE));
	} else {
	  varMemoryMMResult = (uint64_t) (this->m_padded * this->p * sizeof(TYPE));
	  varMemoryMMResult += (uint64_t) (this->m_padded * this->p * sizeof(int));
	}
	
	/* calculate the new required memory */
	requiredMemory = fixedMemory + varMemoryArray + varMemoryMMResult;
      }

      /* determine the tile start positions and the number of rows to compute in a tile */
      this->mTile.resize(this->nTiles);
      fill(this->mTile.begin(), this->mTile.end(), this->m_padded);
      this->mTile.back() = this->m - (this->nTiles-1)*this->m_padded;
      
      this->mStart.resize(this->nTiles);
      fill(this->mStart.begin(), this->mStart.end(), 0); 
      for (int i=1; i<this->nTiles; ++i)
	this->mStart[i] = this->mStart[i-1] + this->mTile[i-1];

#if 0
      for (int i=0; i<this->nTiles; ++i) {
	cout << i+1 << "/" << this->nTiles << " " << this->mStart[i] << " " 
	     << this->mTile[i] << " " << this->mStart[i] + this->mTile[i] << endl;
      }
#endif
    }

    /* stop the timer */
    STOP_TIMER(this->DT,this->start,this->stop,this->dtBuild,KMEANS_ERROR_TILING);
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
    /* start the timer */
    START_TIMER(this->DT,this->start,KMEANS_ERROR_BUILDDATA);

    /* allocate host data */
    this->host_data = new TYPE[this->m*this->n];

    /*****************************/
    /* Randomly shuffle the data */
    /*****************************/

    // obtain a time-based seed:
    std::vector<int> indices(this->m);
    for (int i=0; i<this->m; ++i) indices[i]=i;
    unsigned seed = 0; //std::chrono::system_clock::now().time_since_epoch().count();
    std::shuffle(indices.begin(), indices.end(), std::default_random_engine(seed));

    for (int i=0; i<this->m; ++i) {
      int j = indices[i];
      const TYPE * srcPtr = data + j*this->n;
      TYPE * dstPtr = this->host_data + i*this->n;
      memcpy(dstPtr, srcPtr, sizeof(TYPE)*this->n);
    }

    
    /*******************/
    /* allocate arrays */
    /*******************/

    int mCompute = this->useMiniBatch ? this->m_mb : this->m;
    int mComputePadded = this->useMiniBatch ? this->m_mb_padded : this->m_padded;
    uint64_t nBytes = 0;

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


    /******************************/
    /* array and vector norm data */
    /******************************/

    /* input data */
    int matrixRows = 0;
    int vectorRows = 0;
    if (this->useMiniBatch) {
      if (this->smallMiniBatch) {
	int MM = this->m + this->m_mb;
	matrixRows = ((MM + this->factor - 1) / this->factor)*this->factor;
	vectorRows = matrixRows;
      } else {
	int MM = this->m + this->num_mb_per_tile*this->m_mb;
	matrixRows = ((this->num_mb_per_tile*this->m_mb + this->factor - 1) / this->factor)*this->factor;
	vectorRows = ((MM + this->factor - 1) / this->factor)*this->factor;
      }	
    } else {
      matrixRows = this->m_padded;
      vectorRows = this->m_padded;
    }

    /* input data */
    nBytes = matrixRows * this->n * sizeof(TYPE);
    CUDA_API_SAFE_CALL(cudaMalloc((void**)&(this->dev_data), nBytes), KMEANS_ERROR_BUILDDATA);
    CUDA_API_SAFE_CALL(cudaMemset(this->dev_data, 0, nBytes), KMEANS_ERROR_BUILDDATA);

    /* norm squared of each row of the input data */
    nBytes = vectorRows * sizeof(TYPE);
    CUDA_API_SAFE_CALL(cudaMalloc((void**)&(this->dev_data_norm_squared), nBytes), KMEANS_ERROR_BUILDDATA);
    CUDA_API_SAFE_CALL(cudaMemset(this->dev_data_norm_squared, 0, nBytes), KMEANS_ERROR_BUILDDATA);


    /* allocate space for the result of the matrix matrix multiply */
    if (this->useCUBLAS) {      
      /* matrix matrix multiply buffer */
      nBytes = mComputePadded * this->k * sizeof(TYPE);
      CUDA_API_SAFE_CALL(cudaMalloc((void**)&(this->dev_mmMult), nBytes), KMEANS_ERROR_BUILDDATA);
      CUDA_API_SAFE_CALL(cudaMemset(dev_mmMult, 0, nBytes), KMEANS_ERROR_BUILDDATA);
    } else {      
      /* the partial reduction array */
      nBytes = mComputePadded * this->p * sizeof(TYPE);
      CUDA_API_SAFE_CALL(cudaMalloc((void**)&(this->dev_partial_reduction), nBytes), KMEANS_ERROR_BUILDDATA);
      CUDA_API_SAFE_CALL(cudaMemset(this->dev_partial_reduction, 0, nBytes), KMEANS_ERROR_BUILDDATA);
      
      /* the partial reduction array indices */
      nBytes = mComputePadded * this->p * sizeof(int);
      CUDA_API_SAFE_CALL(cudaMalloc((void**)&(this->dev_partial_reduction_indices), nBytes), KMEANS_ERROR_BUILDDATA);
      CUDA_API_SAFE_CALL(cudaMemset(this->dev_partial_reduction_indices, 0, nBytes), KMEANS_ERROR_BUILDDATA);
    }

    /* stop the timer */
    STOP_TIMER(this->DT,this->start,this->stop,this->dtBuild,KMEANS_ERROR_BUILDDATA);

    return KMEANS_SUCCESS;
  }
  catch (...) {
    return KMEANS_ERROR_BUILDDATA;
  }
}


/* copy a tile to the device */
template<class TYPE>
kmeansErrorStatus kmeans<TYPE>::copyTileToDevice(bool standardApproach) {
  
  try {
    /* start the timer */
    START_TIMER(this->DT,this->start,KMEANS_ERROR_COPY_TILE);

    if (standardApproach || this->smallMiniBatch) {
      /* copy a chunk of the rows to the device */
      if (!this->entireDatasetIsOnDevice) {
	const TYPE * src = this->host_data + this->mStart[this->iTile]*this->n;
	int nBytes = this->mTile[this->iTile]*this->n*sizeof(TYPE);
	CUDA_API_SAFE_CALL(cudaMemcpy(this->dev_data,src,nBytes,cudaMemcpyHostToDevice),
			   KMEANS_ERROR_COPY_TILE);
      }
      if (this->nTiles==1) this->entireDatasetIsOnDevice=true;

      /* copy the "Extra" portion */
      if (this->smallMiniBatch) {
	const TYPE * src = this->dev_data;
	TYPE * dst = this->dev_data + this->m*this->n;
	int nBytes = this->m_mb*this->n*sizeof(TYPE);
	CUDA_API_SAFE_CALL(cudaMemcpy(dst,src,nBytes,cudaMemcpyDeviceToDevice),
			   KMEANS_ERROR_COPY_TILE);
      }
    } else {

      int i = this->m_mb_index;
      int j = this->m_mb_index+1;

      if (!this->mMBWrapTile[i]) {
	/* the matrix data */
	int nBytes = (this->mMBStart[j] - this->mMBStart[i]) * this->n * sizeof(TYPE);
	TYPE * src = this->host_data + this->mMBStart[i] * this->n;
	CUDA_API_SAFE_CALL(cudaMemcpy(this->dev_data, src, nBytes, cudaMemcpyHostToDevice),
			   KMEANS_ERROR_COPY_TILE);
	
      } else {
	/* the matrix data */	
	int nBytes = (this->m - this->mMBStart[i]) * this->n * sizeof(TYPE);
	TYPE * src = this->host_data + this->mMBStart[i] * this->n;
	CUDA_API_SAFE_CALL(cudaMemcpy(this->dev_data, src, nBytes, cudaMemcpyHostToDevice),
			   KMEANS_ERROR_COPY_TILE);
	
	if (this->mMBStart[j]!=0) {
	  nBytes = this->mMBStart[j] * this->n * sizeof(TYPE);
	  TYPE * dst = this->dev_data + (this->m - this->mMBStart[i]) * this->n;
	  CUDA_API_SAFE_CALL(cudaMemcpy(dst, this->host_data, nBytes, cudaMemcpyHostToDevice),
			     KMEANS_ERROR_COPY_TILE);
	}
      }
    }      

    /* stop the timer */
    STOP_TIMER(this->DT,this->start,this->stop,this->dtCopy,KMEANS_ERROR_COPY_TILE);
    return KMEANS_SUCCESS;
  }
  catch (...) {
    return KMEANS_ERROR_COPY_TILE;
  }   
}


/* constructMiniBatch */
template<class TYPE>
kmeansErrorStatus kmeans<TYPE>::constructMiniBatch(int index) {

  try {

    /* start the timer */
    START_TIMER(this->DT,this->start,KMEANS_ERROR_CONSTRUCT_MINIBATCH);

    int ind1 = (this->smallMiniBatch ? this->mMBStart[this->m_mb_index] : 0)
      + index*this->m_mb;

    int ind2 = this->mMBStart[this->m_mb_index] + index*this->m_mb;

    /* set the pointers */
    this->dev_data_mb = this->dev_data + ind1 * this->n;
    this->dev_data_norm_squared_mb = this->dev_data_norm_squared + ind2;

    
    /* stop the timer */
    STOP_TIMER(this->DT,this->start,this->stop,this->dtConstructMiniBatch,KMEANS_ERROR_CONSTRUCT_MINIBATCH);
    return KMEANS_SUCCESS;
  }
  catch (...) {
    return KMEANS_ERROR_CONSTRUCT_MINIBATCH;
  }
}

/* initialize */
template<class TYPE>
kmeansErrorStatus kmeans<TYPE>::initialize(TYPE * result) {

  try {
    kmeansCudaErrorStatus err = NO_ERROR;
    kmeansErrorStatus errK = KMEANS_SUCCESS;

    /* normalize the input data */
    for (int i=0; i<this->nTiles; ++i) {
      /* copy a chunk of the rows to the device */
      this->iTile = i;
      errK = this->copyTileToDevice(true);
      if (errK != KMEANS_SUCCESS) return KMEANS_ERROR_INITIALIZE;

      /* start the timer */
      START_TIMER(this->DT,this->start,KMEANS_ERROR_INITIALIZE);

      /* compute the row normalization */
      err = rowNormalize<TYPE>(this->mStart[this->iTile], this->mTile[this->iTile], this->n,
			       this->dev_data, this->dev_data_norm_squared);
      if (err != NO_ERROR) return KMEANS_ERROR_INITIALIZE;

      /* stop the timer */
      STOP_TIMER(this->DT,this->start,this->stop,this->dtRowNormalize,KMEANS_ERROR_INITIALIZE);
    }


#if 0
    TYPE * temp = new TYPE[this->m];
    CUDA_API_SAFE_CALL(cudaMemcpy(temp,this->dev_data_norm_squared,this->m*sizeof(TYPE),
				  cudaMemcpyDeviceToHost), KMEANS_ERROR_INITIALIZE);
    FILE * fid;
    if (this->useMiniBatch)
      fid = fopen("rowNormMB","wb");
    else
      fid = fopen("rowNorm","wb");

    fwrite(temp,sizeof(TYPE),this->m,fid);
    fclose(fid);
    delete [] temp;
#endif

    /* start the timer */
    START_TIMER(this->DT,this->start,KMEANS_ERROR_INITIALIZE);

    /* copy the "Extra" portion */
    if (this->smallMiniBatch) {
      const TYPE * src = this->dev_data_norm_squared;
      TYPE * dst = this->dev_data_norm_squared + this->m;
      int nBytes = this->num_mb_per_tile*this->m_mb*sizeof(TYPE);
      CUDA_API_SAFE_CALL(cudaMemcpy(dst,src,nBytes,cudaMemcpyDeviceToDevice),
			 KMEANS_ERROR_INITIALIZE);
    }

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
	result[j*this->k + i] = this->host_data[points[i] * this->n + j];

    /* copy the data to the GPU */
    int nBytes = this->k * this->n * sizeof(TYPE);
    CUDA_API_SAFE_CALL(cudaMemcpy(this->dev_centers, result, nBytes, cudaMemcpyHostToDevice),
		       KMEANS_ERROR_INITIALIZE);

    /* normalize the centers data */
    err = colNormalize<TYPE>(this->n, this->k, this->dev_centers,
			     this->dev_centers_norm_squared);
    if (err != NO_ERROR) return KMEANS_ERROR_INITIALIZE;

    /* stop the timer */
    STOP_TIMER(this->DT,this->start,this->stop,this->dtInitialize,KMEANS_ERROR_INITIALIZE);
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
  START_TIMER(this->DT,this->start,KMEANS_ERROR_CLOSEST_CENTERS);

  if (this->useCUBLAS) {
    TYPE one = 1.0;
    TYPE zero = 0.0;
    if (sizeof(TYPE) == 4) {
      int m_start =   this->useMiniBatch ? 0          : this->mStart[this->iTile];
      int m_compute = this->useMiniBatch ? this->m_mb : this->mTile[this->iTile];
      const float * srcData = this->useMiniBatch ? (const float *) this->dev_data_mb : 
	(const float *) this->dev_data;
      const float * srcDataNormSquared = this->useMiniBatch ? 
	(const float *) this->dev_data_norm_squared_mb :
	(const float *) this->dev_data_norm_squared;

      CUBLAS_API_SAFE_CALL(cublasSgemm(this->handle, CUBLAS_OP_N, CUBLAS_OP_N,
				       this->k, m_compute, this->n, (const float *)&one,
				       (const float *) this->dev_centers, this->k,
				       (const float *) srcData, this->n,
				       (const float *)&zero, (float *) this->dev_mmMult, this->k),
			   KMEANS_ERROR_CLOSEST_CENTERS);

      err = rowTransformMinimumF(m_start, m_compute, this->k,
				 (const float *)srcDataNormSquared,
				 (const float *)this->dev_centers_norm_squared,
				 (const float *)this->dev_mmMult,
				 (int *) this->dev_ccindex);

    }
    else {
      int m_start =   this->useMiniBatch ? 0          : this->mStart[this->iTile];
      int m_compute = this->useMiniBatch ? this->m_mb : this->mTile[this->iTile];
      const double * srcData = this->useMiniBatch ? (const double *) this->dev_data_mb : 
	(const double *) this->dev_data;
      const double * srcDataNormSquared = this->useMiniBatch ? 
	(const double *) this->dev_data_norm_squared_mb :
	(const double *) this->dev_data_norm_squared;

      CUBLAS_API_SAFE_CALL(cublasDgemm(this->handle, CUBLAS_OP_N, CUBLAS_OP_N,
				       this->k, m_compute, this->n, (const double *)&one,
				       (const double *) this->dev_centers, this->k,
				       (const double *) srcData, this->n,
				       (const double *)&zero, (double *) this->dev_mmMult, this->k),
			   KMEANS_ERROR_CLOSEST_CENTERS);

      err = rowTransformMinimumD(m_start, m_compute, this->k,
				 (const double *)srcDataNormSquared,
				 (const double *)this->dev_centers_norm_squared,
				 (const double *)this->dev_mmMult,
				 (int *) this->dev_ccindex);
    }
  }
  else {
    int m_start =   this->useMiniBatch ? 0          : this->mStart[this->iTile];
    int m_compute = this->useMiniBatch ? this->m_mb : this->mTile[this->iTile];
    TYPE * srcData = this->useMiniBatch ? this->dev_data_mb : this->dev_data;
    TYPE * srcDataNormSquared = this->useMiniBatch ? this->dev_data_norm_squared_mb 
	: this->dev_data_norm_squared;
    
    err = ClosestCenters<TYPE>(m_start, m_compute, 
			       this->n, srcData, this->k, this->dev_centers,
			       srcDataNormSquared, this->dev_centers_norm_squared,
			       this->p, this->dev_partial_reduction,
			       this->dev_partial_reduction_indices,
			       this->dev_ccindex, this->constantMemSet);
  }

  /* stop the timer */
  STOP_TIMER(this->DT,this->start,this->stop,this->dtClosestCenters,KMEANS_ERROR_CLOSEST_CENTERS);
  if (err != NO_ERROR) return KMEANS_ERROR_CLOSEST_CENTERS;
  return KMEANS_SUCCESS;
}


/* cluster centers */
template<class TYPE>
kmeansErrorStatus kmeans<TYPE>::clusterCenters() {

  kmeansCudaErrorStatus err = NO_ERROR;

  /* start the timer */
  START_TIMER(this->DT,this->start,KMEANS_ERROR_CLUSTER_CENTERS);

  int m_start =   this->useMiniBatch ? 0          : this->mStart[this->iTile];
  int m_compute = this->useMiniBatch ? this->m_mb : this->mTile[this->iTile];
  TYPE * srcData = this->useMiniBatch ? this->dev_data_mb : this->dev_data;

  /* compute the new cluster centers */
  err = ClusterCenters<TYPE>(m_start, m_compute, 
			     this->n, this->k, this->lastTile, srcData,
			     this->dev_ccindex, this->dev_centers_large,
			     this->dev_counts_large, this->dev_centers_transpose,
			     this->dev_counts);

  /* stop the timer */
  STOP_TIMER(this->DT,this->start,this->stop,this->dtClusterCenters,KMEANS_ERROR_CLUSTER_CENTERS);
  if (err != NO_ERROR) return KMEANS_ERROR_CLUSTER_CENTERS;
  return KMEANS_SUCCESS;
}

/* compactness */
template<class TYPE>
kmeansErrorStatus kmeans<TYPE>::compactness() {

  kmeansCudaErrorStatus err = NO_ERROR;
  kmeansErrorStatus errK = KMEANS_SUCCESS;

  if (this->useMiniBatch) {
    /* start the timer */
    START_TIMER(this->DT,this->start,KMEANS_ERROR_COMPACTNESS);
    
    /* compute the compactness */
    err = Compactness<TYPE>(0, this->m_mb, true,
			    this->n, this->k, this->dev_data_mb, this->dev_ccindex,
			    this->dev_centers_transpose, this->dev_compactness);
    
    /* stop the timer */
    STOP_TIMER(this->DT,this->start,this->stop,this->dtCompactness,KMEANS_ERROR_COMPACTNESS);

  } else {

    /* loop over the tiles */
    for (int i=0; i<this->nTiles; ++i) {
      /* copy a chunk of the rows to the device */
      this->iTile = i;
      errK = this->copyTileToDevice(true);
      if (errK != KMEANS_SUCCESS) return errK;
      
      /* start the timer */
      START_TIMER(this->DT,this->start,KMEANS_ERROR_COMPACTNESS);
      
      int m_start =   this->useMiniBatch ? 0          : this->mStart[this->iTile];
      int m_compute = this->useMiniBatch ? this->m_mb : this->mTile[this->iTile];
      TYPE * srcData = this->useMiniBatch ? this->dev_data_mb : this->dev_data;
      
      bool lastTile = false;
      if (i==this->nTiles-1) lastTile = true;

      /* compute the compactness */
      err = Compactness<TYPE>(m_start, m_compute, lastTile,
			      this->n, this->k, srcData, this->dev_ccindex,
			      this->dev_centers_transpose, this->dev_compactness);
      
      /* stop the timer */
      STOP_TIMER(this->DT,this->start,this->stop,this->dtCompactness,KMEANS_ERROR_COMPACTNESS);
    }
  }

  /* start the timer */
  START_TIMER(this->DT,this->start,KMEANS_ERROR_COMPACTNESS);

  /* copy back to the host and finish the calculation */
  CUDA_SAFE_CALL(cudaMemcpy(&this->host_compactness,this->dev_compactness,
  			    sizeof(TYPE),
  			    cudaMemcpyDeviceToHost),ERROR_COMPACTNESS);

  /* stop the timer */
  STOP_TIMER(this->DT,this->start,this->stop,this->dtCompactness,KMEANS_ERROR_COMPACTNESS);

  if (err != NO_ERROR) return KMEANS_ERROR_COMPACTNESS;
  return KMEANS_SUCCESS;
}


/* compactness */
template<class TYPE>
kmeansErrorStatus kmeans<TYPE>::transpose() {

  kmeansCudaErrorStatus err = NO_ERROR;

  /* start the timer */
  START_TIMER(this->DT,this->start,KMEANS_ERROR_TRANSPOSE);

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
  STOP_TIMER(this->DT,this->start,this->stop,this->dtTranspose,KMEANS_ERROR_TRANSPOSE);

  if (err != NO_ERROR) return KMEANS_ERROR_TRANSPOSE;
  return KMEANS_SUCCESS;
}

/* reset */
template<class TYPE>
kmeansErrorStatus kmeans<TYPE>::reset() {

  int nBytes = 0;
  kmeansCudaErrorStatus err = NO_ERROR;

  /* start the timer */
  START_TIMER(this->DT,this->start,KMEANS_ERROR_RESET);

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
  STOP_TIMER(this->DT,this->start,this->stop,this->dtReset,KMEANS_ERROR_RESET);

  if (err != NO_ERROR) return KMEANS_ERROR_RESET;
  return KMEANS_SUCCESS;
}

/* compute */
template<class TYPE>
kmeansErrorStatus kmeans<TYPE>::compute() {

  kmeansErrorStatus err = KMEANS_SUCCESS;
  if (this->useMiniBatch) 
    err = this->computeMiniBatch();
  else
    err = this->computeStandard();
  return err;
}

/* compute */
template<class TYPE>
kmeansErrorStatus kmeans<TYPE>::computeStandard() {


  /* allocate timers */
  try {
    kmeansErrorStatus err = KMEANS_SUCCESS;
    int iters = 0;

    while (iters<this->maxIters && this->relErr>this->criterion) {

      /* compute the closest center */
      err = reset();
      if (err != KMEANS_SUCCESS) return err;

      /* set this to true by default ... assume 1 tile */
      this->lastTile = true;

      /* loop over the tiles */
      for (int i=0; i<this->nTiles; ++i) {
	/* copy a chunk of the rows to the device */
	this->iTile = i;
	
	/* logic for last tile */
	if (this->iTile<this->nTiles-1) this->lastTile=false;
	else this->lastTile=true;

	err = this->copyTileToDevice(true);
	if (err != KMEANS_SUCCESS) return err;

	/* compute the closest center */
	err = closestCenters();
	if (err != KMEANS_SUCCESS) return err;

	/* compute the closest center */
	err = clusterCenters();
	if (err != KMEANS_SUCCESS) return err;
      }


      /* compute the compactness ... loop over the tiles is 
	 internal to this routine */
      err = compactness();
      if (err != KMEANS_SUCCESS) return err;
      
      /* transpose the results */
      err = transpose();
      if (err != KMEANS_SUCCESS) return err;

      /* compute the convergence statistic */
      TYPE compactness_old = this->compactnessScore;
      this->compactnessScore = this->host_compactness;
      this->relErr = fabs(this->compactnessScore - compactness_old) / fabs(this->compactnessScore);
      iters++;
      cout << "iteration " << iters << " relErr = " << this->relErr
	   << " compactness = " << this->compactnessScore << endl;
    }
    return KMEANS_SUCCESS;
  }
  catch (...) {
    return KMEANS_ERROR_COMPUTE;
  }
}



/* compute */
template<class TYPE>
kmeansErrorStatus kmeans<TYPE>::computeMiniBatch() {

  try {
    kmeansErrorStatus err = KMEANS_SUCCESS;
    int iters = 0;
    this->m_mb_index = 0;
    this->iTile = 0;

    while (iters<this->maxIters && this->relErr>this->criterion) {

      /* copy a chunk of the rows to the device */
      err = this->copyTileToDevice(false);
      if (err != KMEANS_SUCCESS) return err;

      /* loop over the tiles */
      for (int i=0; i<this->num_mb_per_tile; ++i) {
	/* set this to true by default ... assume 1 tile */
	this->lastTile = true;

	/* reset the data */
	err = reset();
	if (err != KMEANS_SUCCESS) return err;

	/* construct the minibatch dataset */
	err = constructMiniBatch(i);
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
	this->compactnessScore = this->host_compactness;
	this->relErr = fabs(this->compactnessScore - compactness_old) / fabs(this->compactnessScore);
	iters++;
	cout << "iteration " << iters << "(" << i <<  ") relErr = " << this->relErr
	     << " compactness = " << this->compactnessScore << endl;
	if (this->relErr<=this->criterion) return KMEANS_SUCCESS;
      }
      this->m_mb_index++;
    }
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
						      float * result, const float miniBatchFraction) {

  kmeansErrorStatus err = KMEANS_SUCCESS;

  /* Allocate an instance of kmeans */
  kmeans<float> * KMEANS = new kmeans<float>(m, n, k, criterion, maxIters, numRetries,
					     initAlgorithm, useCUBLAS, miniBatchFraction);

  /* compute the tiling */
  err = KMEANS->computeTiling();
  if (err != KMEANS_SUCCESS) goto cleanup;

  /* build the device data */
  err = KMEANS->buildData(data);
  if (err != KMEANS_SUCCESS) goto cleanup;

  /* initialize */
  err = KMEANS->initialize(result);
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
						      double * result, const double miniBatchFraction) {

  kmeansErrorStatus err = KMEANS_SUCCESS;

  /* Allocate an instance of kmeans */
  kmeans<double> * KMEANS = new kmeans<double>(m, n, k, criterion, maxIters, numRetries,
					       initAlgorithm, useCUBLAS, miniBatchFraction);

  /* compute the tiling */
  err = KMEANS->computeTiling();
  if (err != KMEANS_SUCCESS) goto cleanup;

  /* build the device data */
  err = KMEANS->buildData(data);
  if (err != KMEANS_SUCCESS) goto cleanup;

  /* initialize */
  err = KMEANS->initialize(result);
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
