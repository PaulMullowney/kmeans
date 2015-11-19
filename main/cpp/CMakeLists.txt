##
# Build the shared library
##
INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS}) 
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}) 

INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/main/cuda) 
LINK_DIRECTORIES(${PROJECT_BINARY_DIR}/main/cuda) 

INCLUDE_DIRECTORIES(${MAGMA_INCLUDE_DIRS}) 
#LINK_DIRECTORIES(${MAGMA_LIBRARY_DIRS}) 

SET(MAGMA_LIBRARY ${MAGMA_LIBRARY_DIRS}/libmagma.a)
MESSAGE(${MAGMA_LIBRARY})

# source files to compile with GCC/G++ ... usually these contain the C-Wrapper
SET(CPP_Sources kmeans.cpp)

# Build the base library
CUDA_ADD_LIBRARY(kmeans SHARED ${CPP_Sources})
CUDA_ADD_CUBLAS_TO_TARGET(kmeans)
#SET_TARGET_PROPERTIES(kmeans PROPERTIES LINKER_LANGUAGE C)
TARGET_LINK_LIBRARIES(kmeans ${KMEANS_CUDA_LIBRARY} ${MAGMA_LIBRARY})
SET(KMEANS_CPP_LIBRARY kmeans PARENT_SCOPE)

# install to the destination in the lib subdir.
# later we'll put include, docs, share, ... as the library grows.
INSTALL(TARGETS kmeans COMPONENT libraries DESTINATION lib)
INSTALL(FILES kmeans.hpp DESTINATION include)
