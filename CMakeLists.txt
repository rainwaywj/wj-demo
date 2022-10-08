#wj-demo

cmake_minimum_required(VERSION 3.5)

project(wj-demo)

option(LINK_SHARE_LIBRARY "link share library" ON)

find_package(Threads REQUIRED)

set(PROJECT_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)

set(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib)

include_directories(${PROJECT_ROOT_DIR}/include)

link_directories(${PROJECT_ROOT_DIR}/lib)

set(CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -std=c++11)

#deubg option
#set(CUDA_NVCC_FLAGS_DEBUG "-g -G")
#set(CUDA_CXX_FLAGS_DEBUG "-g")

#release options
#set(CUDA_NVCC_FLAGS_RELEASE "-g -G")
#set(CUDA_CXX_FLAGS_RELEASE "-g")

add_subdirectory(src)
add_subdirectory(thirdparty)
add_subdirectory(test)
add_subdirectory(examples)
# add_subdirectory(tools)