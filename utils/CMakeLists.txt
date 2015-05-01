##
# Some utility files to be configured
##
SET(batchJobGPUin          ${CMAKE_CURRENT_SOURCE_DIR}/batchJobGPU.in)
SET(batchJobGPU            ${PROJECT_BINARY_DIR}/batchJobGPU)  
CONFIGURE_FILE(${batchJobGPUin} ${batchJobGPU} @ONLY )	

SET(env_vars_in            ${CMAKE_CURRENT_SOURCE_DIR}/env_vars.sh.in)
SET(env_vars               ${PROJECT_BINARY_DIR}/env_vars.sh)  
CONFIGURE_FILE(${env_vars_in} ${env_vars} @ONLY )	

INSTALL(FILES ${batchJobGPU} COMPONENT utilities DESTINATION bin)
INSTALL(FILES ${env_vars} COMPONENT utilities DESTINATION tests)
