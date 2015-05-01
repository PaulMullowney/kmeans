##
# Testing setup
##

# adds test target to Makefile or RUN_TESTS for MSVC
# Its annoying that this seems to be needed in here
# and in the top level parent directory??
ENABLE_TESTING()

INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}) 

INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/test/gtest-1.7.0/include)
LINK_DIRECTORIES(${PROJECT_BINARY_DIR}/test/gtest-1.7.0/) 
SET(GTEST_LIBRARY libgtest.so)

INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/main/cuda) 
LINK_DIRECTORIES(${PROJECT_SOURCE_DIR}/main/cuda) 

# CUDA
SET(CUDA_KERNEL_LIBRARY)
if (${CUDA_FOUND})
SET(CUDA_KERNEL_LIBRARY ${CUDA_KERNEL_LIBRARY} CudaKernelLibrary)
endif()

# Add a test for simpleTest
#CUDA_ADD_EXECUTABLE(simpleTest simpleTest.c)
#TARGET_LINK_LIBRARIES(simpleTest ${CUDA_KERNEL_LIBRARY})
#ADD_TEST(SIMPLETEST simpleTest)
#INSTALL(TARGETS simpleTest COMPONENT tests DESTINATION tests)
