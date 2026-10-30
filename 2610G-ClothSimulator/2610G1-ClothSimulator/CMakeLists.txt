cmake_minimum_required(VERSION 3.20)

if (WIN32)
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
    add_compile_definitions(_USE_MATH_DEFINES)
endif()

project(ClothSimulator LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

include_directories(src)
file(GLOB_RECURSE SOURCE "src/*.cu" "src/*.cuh")
add_executable(ClothSimulator ${SOURCE})
target_compile_options(ClothSimulator PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=20015,177,20011,20012,20013,20014>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-g;-G;-O0>
)

# OpenGL
find_package(OpenGL REQUIRED)

# GLAD
add_subdirectory(lib/glad)

# GLFW
find_package(glfw3 REQUIRED)

# jsoncpp
if (WIN32)
    find_package(jsoncpp CONFIG REQUIRED)
else()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(JSONCPP REQUIRED jsoncpp)
endif()

# Eigen
find_package(Eigen3 REQUIRED)

# CUDA
find_package(CUDA REQUIRED)
get_filename_component(CUDA_RUNTIME_LIBRARY_DIR "${CUDA_cudart_static_LIBRARY}" DIRECTORY)
# Prefer this toolkit's runtime over older CUDA libraries in system directories.
find_library(CUDA_DEVICE_RUNTIME_LIBRARY cudadevrt
    PATHS "${CUDA_RUNTIME_LIBRARY_DIR}" NO_DEFAULT_PATH REQUIRED)
target_link_libraries(ClothSimulator "${CUDA_DEVICE_RUNTIME_LIBRARY}")

set_target_properties(ClothSimulator PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
include_directories(
    ${OPENGL_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
)
if (WIN32)
    target_link_libraries(
        ClothSimulator
        glad
        glfw
        JsonCpp::JsonCpp
        ${CUDA_cublas_LIBRARY}
        ${CUDA_cusparse_LIBRARY}
    )
else()
    target_include_directories(ClothSimulator PRIVATE ${JSONCPP_INCLUDE_DIRS})
    target_link_directories(ClothSimulator PRIVATE ${JSONCPP_LIBRARY_DIRS})
    target_link_libraries(
        ClothSimulator
        glad
        glfw
        ${JSONCPP_LIBRARIES}
        ${CUDA_cublas_LIBRARY}
        ${CUDA_cusparse_LIBRARY}
    )
endif()
