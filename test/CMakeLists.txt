# Subdirectories
IF (WIN32)
ELSE()
  ADD_SUBDIRECTORY(gtest-1.7.0)
ENDIF()

if (CUDA_FOUND)
ADD_SUBDIRECTORY(cuda)
endif()
ADD_SUBDIRECTORY(cpp)

ADD_SUBDIRECTORY(python)


# adds test target to Makefile or RUN_TESTS for MSVC
# Its annoying that this seems to be needed in here
# and in the top level parent directory??
ENABLE_TESTING()
