cmake_minimum_required(VERSION 3.25)

# Enforce C++ compiler for CUDA
# Check if /usr/bin/g++-12 exists
execute_process(
    COMMAND /usr/bin/g++-12 --version
    RESULT_VARIABLE GPP_VERSION_CHECK_RESULT
    OUTPUT_VARIABLE GPP_VERSION_OUTPUT
    ERROR_QUIET
)

# If g++-12 exists, set CMAKE_CUDA_HOST_COMPILER
if(GPP_VERSION_CHECK_RESULT EQUAL 0)
    set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/g++-12")
else()
    message(STATUS "g++-12 not found, not setting CMAKE_CUDA_HOST_COMPILER")
endif()

if(GPP_VERSION_CHECK_RESULT EQUAL 0)
    set(CMAKE_CXX_COMPILER "/usr/bin/g++-12")
else()
    message(STATUS "g++-12 not found, not setting CMAKE_CXX_COMPILER")
endif()

# Specify the maximum number of concurrent processes for building
set(CMAKE_BUILD_PARALLEL_LEVEL 8)

# specify architecture for rtx 4060
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "86;89")
endif()

# get current folder name
string(REGEX REPLACE ".*/(.*)" "\\1" CURRENT_FOLDER ${CMAKE_CURRENT_SOURCE_DIR})

project("coherent_dedispersion")
enable_language(CUDA)

set(MY_INCLUDE_DIRS
  ${PROJECT_SOURCE_DIR}/utils/include
  ${PROJECT_SOURCE_DIR}/psrdada/include
  ${PROJECT_SOURCE_DIR}/folding/include
  ${PROJECT_SOURCE_DIR}/gpu/include
  ${PROJECT_SOURCE_DIR}/dedispersion/include
  ${PROJECT_SOURCE_DIR}/simulated_pulsar/include
)

set(MY_SOURCE_CPP_FILES
  ${PROJECT_SOURCE_DIR}/utils/*.cpp
  ${PROJECT_SOURCE_DIR}/psrdada/*.cpp
  ${PROJECT_SOURCE_DIR}/folding/*.cpp
  ${PROJECT_SOURCE_DIR}/dedispersion/msosm/*.cpp
  ${PROJECT_SOURCE_DIR}/dedispersion/msosm_dm_loop/*.cpp
  ${PROJECT_SOURCE_DIR}/dedispersion/msosm_stream/*.cpp
  ${PROJECT_SOURCE_DIR}/dedispersion/msosm_dm_stream/*.cpp
  ${PROJECT_SOURCE_DIR}/dedispersion/msosm_dm_concurrent/*.cpp
  ${PROJECT_SOURCE_DIR}/dedispersion/overlap_save/*.cpp
  ${PROJECT_SOURCE_DIR}/dedispersion/overlap_save_dm_concurrent/*.cpp
  ${PROJECT_SOURCE_DIR}/simulated_pulsar/*.cpp
)

set(MY_SOURCE_CU_FILES
  ${PROJECT_SOURCE_DIR}/gpu/*.cu
)

include_directories(${MY_INCLUDE_DIRS})

# file
file(GLOB CPP_FILES ${MY_SOURCE_CPP_FILES})
file(GLOB CU_FILES ${MY_SOURCE_CU_FILES})

# List of executables
set(EXECTABLES
  msosm_psrdada
  multiple_thread_osm
  multiple_stream_msosm
  multiple_dm_stream_msosm
  multiple_dm_loop_msosm
  multiple_dm_concurrent_msosm
  multiple_dm_concurrent_osm
  test_processing_speed_msosm
  test_processing_speed_osm
  test_processing_speed_msosm_batch32_dm0to100
  test_processing_speed_osm_batch1_dm0to100
  test_processing_speed_msosm_dmtrial
  test_processing_speed_osm_dmtrial
  test_multiple_dm_msosm_fused
  test_multiple_dm_msosm_stream
  check_msosm
  # Add more executables here
)

# cuFFT
find_package(CUDAToolkit REQUIRED)
# cuFFTDx
# find_package(mathdx REQUIRED COMPONENTS cufftdx CONFIG PATHS "/home/joe/GPU/nvidia-mathdx-24.01.0/nvidia/mathdx/24.01/")
# matplotlib-cpp (installed by vcpkg)
find_package(Python COMPONENTS Development NumPy)
find_path(MATPLOTLIB_CPP_INCLUDE_DIRS "matplotlibcpp.h")

foreach(EXE ${EXECTABLES})

  # target
  add_executable(${EXE} ${EXE}.cpp ${CPP_FILES} ${CU_FILES})
  target_include_directories(${EXE} PRIVATE ${MY_INCLUDE_DIRS})

  # cuFFT
  target_link_libraries(${EXE} CUDA::cufft)

  # cuFFTDx
  # target_link_libraries(${EXE} mathdx::cufftdx)

  # matplotlib-cpp (installed by vcpkg)
  target_link_libraries(${EXE} Python::Python Python::NumPy)

  target_include_directories(${EXE} PRIVATE ${MATPLOTLIB_CPP_INCLUDE_DIRS})

  # FFTW
  target_link_libraries(${EXE} fftw3f)

  #NVTXtest_processing_speed_msosm.cpp
  target_link_libraries(${EXE} CUDA::nvtx3)

endforeach()