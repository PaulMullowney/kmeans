# simple cmake file for standalone benchmarks
cmake_minimum_required(VERSION 2.8.11 FATAL_ERROR)
PROJECT(kmeans-gpu)

SET(kmeansGPULibrary_VERSION_MAJOR 0)
SET(kmeansGPULibrary_VERSION_MINOR 0)
SET(kmeansGPULibrary_VERSION_PATCH 1)

INCLUDE (${CMAKE_ROOT}/Modules/CheckIncludeFile.cmake)
#INCLUDE (${CMAKE_ROOT}/Modules/CheckTypeSize.cmake)
#INCLUDE (${CMAKE_ROOT}/Modules/CheckFunctionExists.cmake)
#INCLUDE (${CMAKE_ROOT}/Modules/CheckCXXSourceCompiles.cmake)
#INCLUDE (${CMAKE_ROOT}/Modules/TestBigEndian.cmake)
#INCLUDE (${CMAKE_ROOT}/Modules/CheckSymbolExists.cmake)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -Wall -std=c++11")
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -g -Wall")


#########
# CUDA
#########
OPTION(WITH_CUDA "Build the library with CUDA" ON)
IF (WITH_CUDA) 
  #SET(CUDA_TOOLKIT_ROOT_DIR /cygdrive/c/CUDA_6.5/)
  #SET(CUDA_SDK_ROOT_DIR /cygdrive/c/CUDA_6.5/)
  FIND_PACKAGE(CUDA REQUIRED)
  IF(CUDA_FOUND)
    CHECK_INCLUDE_FILE(${CUDA_INCLUDE_DIRS}/cuda_runtime.h HAVE_CUDA_H)

    OPTION(CUDA_ARCH_52 "Build the library under sm_52 only." ON)
    OPTION(CUDA_ARCH_50 "Build the library under sm_50 only." OFF)
    OPTION(CUDA_ARCH_35 "Build the library under sm_35 only." OFF)
    OPTION(CUDA_ARCH_30 "Build the library under sm_30 only." OFF)

    IF(CUDA_ARCH_52)
	SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-O3 -g -lineinfo -Xptxas -v -std=c++11 -arch=sm_52")
    ELSEIF(CUDA_ARCH_50)
	SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-O3 -g -Xptxas -v -std=c++11 -arch=sm_50")
    ELSEIF(CUDA_ARCH_35)
	SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-O3 -g -Xptxas -v -std=c++11 -arch=sm_35")
    ELSEIF(CUDA_ARCH_30)
	SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-O3 -g -Xptxas -v -arch=sm_30")
    ELSE()
        SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-O3 -g -Xptxas -v -std=c++11 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52")
    ENDIF()
  ENDIF()
ENDIF()

#########
# OPENMP 
#########
FIND_PACKAGE(OpenMP REQUIRED)
IF (OPENMP_FOUND)
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
ENDIF()

################
# DOCUMENTATION
################
OPTION(BUILD_DOCUMENTATION "Create and install the HTML based API documentation (requires Doxygen)" OFF)
IF(BUILD_DOCUMENTATION) 
  FIND_PACKAGE(Doxygen)
  IF(NOT DOXYGEN_FOUND)
    MESSAGE(FATAL_ERROR
      "Doxygen is needed to build the documentation.")
  ENDIF()

  SET(doxyfile_in          ${CMAKE_CURRENT_SOURCE_DIR}/doxyfile.dox.in )
  SET(doxyfile             ${PROJECT_BINARY_DIR}/Doxyfile              )
  SET(doxy_html_index_file ${CMAKE_CURRENT_BINARY_DIR}/html/index.html )
  SET(doxy_output_root     ${CMAKE_CURRENT_BINARY_DIR}                 )
 
  CONFIGURE_FILE(${doxyfile_in} ${doxyfile} @ONLY )
 
  ADD_CUSTOM_COMMAND(OUTPUT ${doxy_html_index_file}
                     COMMAND ${DOXYGEN_EXECUTABLE} ${doxyfile}
                     MAIN_DEPENDENCY ${doxyfile} ${doxyfile_in}
                     COMMENT "Generating HTML documentation")
 
  ADD_CUSTOM_TARGET( doc ALL DEPENDS ${doxy_html_index_file} )
 
  INSTALL(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html COMPONENT documentation DESTINATION share/doc)
ENDIF()

################
# CTAGS
################
OPTION(BUILD_TAGS "Build Tags" OFF)
IF(BUILD_TAGS) 
  SET_SOURCE_FILES_PROPERTIES(ctags.output PROPERTIES GENERATED true)
  ADD_CUSTOM_COMMAND(OUTPUT ctags.output
    COMMAND ctags -e -R --exclude=java . --langmap=c++:+.cu *
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
  ADD_CUSTOM_TARGET(CTagsOutput ALL DEPENDS ctags.output)
ENDIF()

####################
# Generate config.h
####################
configure_file ("${PROJECT_SOURCE_DIR}/config.h.in"
                "${PROJECT_BINARY_DIR}/config.h" )
include_directories(${PROJECT_BINARY_DIR})


#################
# Subdirectories
#################
ADD_SUBDIRECTORY(main)
ADD_SUBDIRECTORY(test)
ADD_SUBDIRECTORY(utils)

# adds test target to Makefile or RUN_TESTS for MSVC
# Its annoying that this seems to be needed in here
# and in the test subdirectory??
ENABLE_TESTING()

#################
# CPack
#################
SET(CPACK_PACKAGE_DESCRIPTION_SUMMARY "KMeans Clustering GPU Package")
SET(CPACK_PACKAGE_VENDOR "DigitalGlobe, Inc.")
SET(CPACK_SOURCE_GENERATOR "TGZ")
SET(CPACK_PACKAGE_VERSION_MAJOR ${kmeansGPULibrary_VERSION_MAJOR})
SET(CPACK_PACKAGE_VERSION_MINOR ${kmeansGPULibrary_VERSION_MINOR})
SET(CPACK_PACKAGE_VERSION_PATCH ${kmeansGPULibrary_VERSION_PATCH})
SET(CPACK_PACKAGE_FILE_NAME "${CMAKE_PROJECT_NAME}-${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}")
SET(CPACK_ARCHIVE_COMPONENT_INSTALL ON) 
SET(CPACK_COMPONENTS_ALL executables libraries)
SET(CPACK_COMPONENTS_GROUPING ALL_COMPONENTS_IN_ONE)
SET(CPACK_COMPONENT_executables_DISPLAY_NAME "Kmeans Clustering Executables")
SET(CPACK_COMPONENT_executables_DESCRIPTION "Executables for Kmeans Clustering")
SET(CPACK_COMPONENT_libraries_DISPLAY_NAME "Kmeans Clustering Libraries")
SET(CPACK_COMPONENT_libraries_DESCRIPTION "Librarries for Kmeans Clustering")
INCLUDE(CPack)
