cmake_minimum_required(VERSION 3.24)
if(NOT DEFINED CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX})
    find_program(CMAKE_CUDA_COMPILER NAMES nvcc REQUIRED)
endif()
project(CUDAClothXPBD LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(Boost REQUIRED COMPONENTS chrono thread program_options)

add_executable(CUDAClothXPBD
    main.cu
    src/cloth.cu
    src/kernels.cu
    src/vec3.cu
    src/camera.cpp
    src/state.cpp
    src/utils.cpp
)
target_include_directories(CUDAClothXPBD PRIVATE inc)
target_compile_features(CUDAClothXPBD PRIVATE cxx_std_17 cuda_std_17)
set_target_properties(CUDAClothXPBD PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(CUDAClothXPBD PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G>
)
target_link_libraries(CUDAClothXPBD PRIVATE
    CUDA::cudart OpenGL::GL OpenGL::GLU GLUT::GLUT GLEW::GLEW
    glm::glm Boost::chrono Boost::thread Boost::program_options
)
