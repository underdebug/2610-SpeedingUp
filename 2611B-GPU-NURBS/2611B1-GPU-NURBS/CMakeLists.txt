cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND "$ENV{CUDAARCHS}" STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures")
endif()

# Detect CUDA before CXX so CMake searches PATH before the host compiler directory.
project(NURBS LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Prefer libraries from the CUDA compiler's toolkit over system installations.
if(NOT DEFINED CUDAToolkit_ROOT AND "$ENV{CUDAToolkit_ROOT}" STREQUAL "")
    set(CUDAToolkit_ROOT "${CMAKE_CUDA_COMPILER_TOOLKIT_ROOT}")
endif()
find_package(CUDAToolkit REQUIRED)

add_executable(NURBS
    main.cu
    Evaluate.cu
    ExtraMath.cu
    Matrix.cu
    Operator.cu
    kernel.cu
    test.cu
)

target_compile_definitions(NURBS PRIVATE FIX1)
target_compile_options(NURBS PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-dlcm=ca>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-g>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-O0>
)
target_link_libraries(NURBS PRIVATE CUDA::cublas CUDA::cusparse)
get_target_property(CUDA_CUBLAS_PATH CUDA::cublas IMPORTED_LOCATION)
get_filename_component(CUDA_LIBRARY_DIR "${CUDA_CUBLAS_PATH}" DIRECTORY)
target_link_options(NURBS PRIVATE "$<DEVICE_LINK:-L${CUDA_LIBRARY_DIR}>")
set_target_properties(NURBS PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
