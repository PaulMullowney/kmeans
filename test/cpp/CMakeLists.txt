##
# Testing setup
##

# adds test target to Makefile or RUN_TESTS for MSVC
# Its annoying that this seems to be needed in here
# and in the top level parent directory??
ENABLE_TESTING()

INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}) 

IF (WIN32)
SET(GTEST_LIBRARY )
ELSE()
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/test/gtest-1.7.0/include)
LINK_DIRECTORIES(${PROJECT_BINARY_DIR}/test/gtest-1.7.0/) 
#SET(GTEST_LIBRARY libgtest.so)
SET(GTEST_LIBRARY gtest)
ENDIF()

INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/main/cuda) 
LINK_DIRECTORIES(${PROJECT_BINARY_DIR}/main/cuda) 
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/main/cpp) 
LINK_DIRECTORIES(${PROJECT_BINARY_DIR}/main/cpp) 

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -Wall")
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -g -Wall")

# add definitions if in debug mode
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
   ADD_DEFINITIONS(-DPRINT_DEBUG=1)
endif()

ADD_DEFINITIONS(-DTIMER_PROFILE=1)

if (CUDA_FOUND)
   # Add a test for Kmeans
   CUDA_ADD_EXECUTABLE(KmeansTests KmeansTests.cpp)
   TARGET_LINK_LIBRARIES(KmeansTests ${KMEANS_CPP_LIBRARY} ${KMEANS_CUDA_LIBRARY} ${GTEST_LIBRARY})
   ADD_TEST(KMEANS_SIFT        KmeansTests mat-sift 898790 128 256 50 0 1.e-5)
   ADD_TEST(KMEANS_SIFT_CUBLAS KmeansTests mat-sift 898790 128 256 50 1 1.e-5)
   ADD_TEST(KMEANS_HOG         KmeansTests mat-hog  796160 324 256 50 0 1.e-5)
   ADD_TEST(KMEANS_HOG_CUBLAS  KmeansTests mat-hog  796160 324 256 50 1 1.e-5)
   INSTALL(TARGETS KmeansTests COMPONENT tests DESTINATION tests)

   # Add a test for MatMult
   CUDA_ADD_EXECUTABLE(MatMultTests MatMultTests.cpp)
   TARGET_LINK_LIBRARIES(MatMultTests ${KMEANS_CPP_LIBRARY} ${KMEANS_CUDA_LIBRARY} ${GTEST_LIBRARY})
   ADD_TEST(MATMULTTESTS MatMultTests)
   INSTALL(TARGETS MatMultTests COMPONENT tests DESTINATION tests)

endif()

#file(COPY "${CMAKE_SOURCE_DIR}/test/cpp/sampleRF.txt" DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
#INSTALL(FILES "${CMAKE_SOURCE_DIR}/test/cpp/sampleRF.txt" COMPONENT tests DESTINATION tests)
