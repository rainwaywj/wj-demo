# test-cuda

set(TARGET_NAME test-gpu-bandwidth)

set(SOURCE_LIST
        test-gpu-bandwidth.cpp
)

add_executable(
        ${TARGET_NAME}
        ${SOURCE_LIST}
)
target_include_directories(
        ${TARGET_NAME}
        PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/src        
)

target_link_libraries(
  ${TARGET_NAME}
  PRIVATE cuda_suite
  PUBLIC Threads::Threads)
