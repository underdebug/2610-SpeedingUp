cmake_minimum_required(VERSION 3.18)

project(NURBS LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
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

add_executable(NrcTest Evaluate.cu_.cu Evaluate.cu ExtraMath.cu Matrix.cu Operator.cu)
target_link_libraries(NrcTest PRIVATE CUDA::cublas CUDA::cusparse)
target_link_options(NrcTest PRIVATE "$<DEVICE_LINK:-L${CUDA_LIBRARY_DIR}>")
set_target_properties(NrcTest PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
