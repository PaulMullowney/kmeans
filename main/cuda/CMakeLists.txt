##
# Build the shared library
## 

## Don't really need these right now, but good to see an example
#FIND_CUDA_HELPER_LIBS(curand)
#SET(CUDA_LIBRARIES ${CUDA_LIBRARIES} ${CUDA_curand_LIBRARY})
#FIND_CUDA_HELPER_LIBS(cusparse)
#SET(CUDA_LIBRARIES ${CUDA_LIBRARIES} ${CUDA_cusparse_LIBRARY})

INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}) 

# source files to compile with NVCC ... usually these contain the C-Wrapper
SET(CUDA_Sources MatMatMult.cu ClosestCenters.cu ClusterCenters.cu Compactness.cu RowTransformMinimum.cu Normalize.cu)

INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/main/cuda) 
LINK_DIRECTORIES(${PROJECT_BINARY_DIR}/main/cuda) 

#INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/main/cpp) 
#LINK_DIRECTORIES(${PROJECT_BINARY_DIR}/main/cpp) 

# Build into a .so ... with C linkage
CUDA_ADD_LIBRARY(CudaKernelLibrary SHARED ${CUDA_Sources})
SET_TARGET_PROPERTIES(CudaKernelLibrary PROPERTIES LINKER_LANGUAGE C)

SET(KMEANS_CUDA_LIBRARY CudaKernelLibrary PARENT_SCOPE)

# install to the destination in the lib subdir.
# later we'll put include, docs, share, ... as the library grows.
INSTALL(TARGETS CudaKernelLibrary COMPONENT libraries DESTINATION lib)
