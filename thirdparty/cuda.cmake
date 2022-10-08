include(ExternalProject)
if(NOT BUILD_CUDA_VERSION)
  set(BUILD_CUDA_VERSION
      "10.1"
      CACHE STRING "build cuda version: 10.1")
else(NOT BUILD_CUDA_VERSION)
  set(BUILD_CUDA_VERSION
      ${BUILD_CUDA_VERSION}
      CACHE STRING "build cuda version: 10.1")
endif(NOT BUILD_CUDA_VERSION)
message("BUILD_CUDA_VERSION = ${BUILD_CUDA_VERSION}")

find_path(LIBCUDART_LINK_PATH libcudart_static.a
          /usr/local/cuda-${BUILD_CUDA_VERSION}/lib64)
if(NOT LIBCUDART_LINK_PATH)
  message(FATAL_ERROR "ERROR: No support cudart ${BUILD_CUDA_VERSION}")
else(NOT LIBCUDART_LINK_PATH)
  set(LIBCUDART_FILENAME ${LIBCUDART_LINK_PATH}/libcudart_static.a)
endif(NOT LIBCUDART_LINK_PATH)
# set(LIBCUDART_FILENAME
# "/usr/local/cuda-${BUILD_CUDA_VERSION}/lib64/libcudart_static.a")
# get_filename_component(LIBCUDART_LINK_PATH ${LIBCUDART_FILENAME} DIRECTORY)
get_filename_component(LIBCUDART_PATH ${LIBCUDART_LINK_PATH} DIRECTORY)
get_filename_component(LIBCUDART_PATH "${LIBCUDART_PATH}" REALPATH)
message("LIBCUDART_PATH = ${LIBCUDART_PATH}")
message("LIBCUDART_LINK_PATH = ${LIBCUDART_LINK_PATH}")

add_library(cudart INTERFACE IMPORTED GLOBAL)
target_link_libraries(
  cudart
  INTERFACE ${LIBCUDART_FILENAME}
  INTERFACE ${CMAKE_DL_LIBS}
  INTERFACE rt)
set_target_properties(cudart PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                                        ${LIBCUDART_PATH}/include)
message("cuda include path: ${LIBCUDART_PATH}/include")

ExternalProject_Add(
            nvencode_build
            URL http://distribute.sng.megvii-inc.com/meepo/Video_Codec_SDK_10.0.26.zip
            URL_HASH SHA1=a257e9afaa1b2463d9cc02f8c09714933f990449
            DOWNLOAD_NAME Video_Codec_SDK_10.0.26.zip
            # URL http://distribute.sng.megvii-inc.com/pkgs/Video_Codec_SDK_9.0.20.zip
            # URL_HASH SHA1=dcaae1caf764cc516ff8819ad8bf53d4121393fb
            # DOWNLOAD_NAME Video_Codec_SDK_9.0.20.zip
            DOWNLOAD_DIR ${PKG_CACHE_PATH}
            BUILD_IN_SOURCE 1
            CONFIGURE_COMMAND ""
            BUILD_COMMAND ""
            INSTALL_COMMAND ""
            BUILD_BYPRODUCTS
            <SOURCE_DIR>/Interface/nvEncodeAPI.h
            <SOURCE_DIR>/Lib/linux/stubs/x86_64/libnvidia-encode.so
    )
    ExternalProject_Get_Property(nvencode_build source_dir)
    file(MAKE_DIRECTORY "${source_dir}/Interface")
    add_library(nvencode SHARED IMPORTED GLOBAL)
    add_dependencies(nvencode nvencode_build)
    set_target_properties(nvencode PROPERTIES
        IMPORTED_LOCATION "${source_dir}/Lib/linux/stubs/x86_64/libnvidia-encode.so"
        INTERFACE_INCLUDE_DIRECTORIES "${source_dir}/Interface"
    )
    set(LIB_NVIDIA_ENCODE_FILE "${source_dir}/Lib/linux/stubs/x86_64/libnvidia-encode.so")


add_library(cuda_suite INTERFACE IMPORTED GLOBAL)
if(${BUILD_CUDA_VERSION} STREQUAL "8.0")
  target_link_libraries(
    cuda_suite
    INTERFACE ${LIBCUDART_LINK_PATH}/libnppi_static.a
    INTERFACE ${LIBCUDART_LINK_PATH}/libnppc_static.a
    INTERFACE ${LIBCUDART_LINK_PATH}/libcublas_static.a
    INTERFACE ${LIBCUDART_LINK_PATH}/libculibos.a
    INTERFACE cudart)
elseif(${BUILD_CUDA_VERSION} STREQUAL "9.2")
  target_link_libraries(
    cuda_suite
    INTERFACE ${LIBCUDART_LINK_PATH}/libnppicc_static.a
    INTERFACE ${LIBCUDART_LINK_PATH}/libnppicom_static.a
    INTERFACE ${LIBCUDART_LINK_PATH}/libnppif_static.a
    INTERFACE ${LIBCUDART_LINK_PATH}/libnppig_static.a
    INTERFACE ${LIBCUDART_LINK_PATH}/libnppim_static.a
    INTERFACE ${LIBCUDART_LINK_PATH}/libnppisu_static.a
    INTERFACE ${LIBCUDART_LINK_PATH}/libnppc_static.a
    INTERFACE ${LIBCUDART_LINK_PATH}/libcublas_static.a
    INTERFACE ${LIBCUDART_LINK_PATH}/libculibos.a
    INTERFACE cudart)
elseif(${BUILD_CUDA_VERSION} STREQUAL "10.1")
  target_link_libraries(
    cuda_suite
    INTERFACE ${LIBCUDART_LINK_PATH}/libnppist_static.a
    INTERFACE ${LIBCUDART_LINK_PATH}/libnppitc_static.a
    INTERFACE ${LIBCUDART_LINK_PATH}/libnpps_static.a
    INTERFACE ${LIBCUDART_LINK_PATH}/libnppicc_static.a
    INTERFACE ${LIBCUDART_LINK_PATH}/libnppicom_static.a
    INTERFACE ${LIBCUDART_LINK_PATH}/libnppif_static.a
    INTERFACE ${LIBCUDART_LINK_PATH}/libnppig_static.a
    INTERFACE ${LIBCUDART_LINK_PATH}/libnppim_static.a
    INTERFACE ${LIBCUDART_LINK_PATH}/libnppisu_static.a
    INTERFACE ${LIBCUDART_LINK_PATH}/libnppial_static.a
    INTERFACE ${LIBCUDART_LINK_PATH}/libnppc_static.a
    INTERFACE ${LIBCUDART_LINK_PATH}/libnppidei_static.a
    INTERFACE ${LIBCUDART_LINK_PATH}/libnvjpeg_static.a
    INTERFACE ${LIBCUDART_LINK_PATH}/libculibos.a
    INTERFACE ${LIBCUDART_LINK_PATH}/libcudadevrt.a
    INTERFACE cudart
    INTERFACE nvencode
    )
else()
  message(FATAL_ERROR "Creat cuda_suite failed")
endif()
