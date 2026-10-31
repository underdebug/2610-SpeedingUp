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
    $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=20015,177,20011,20012,20013,20014,940>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:GNU,Clang>>:-Xcompiler=-Wno-return-type>
    # Host debugging only; omit -G to retain device optimization.
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-g;-O0>
)

# Enable GPU kernel debugging only for the listed .cu files in Debug builds.
# Uncomment the files you want to debug; paths are relative to this directory.
# MINGJIE
set(CUDA_DEVICE_DEBUG_SOURCES
    src/Mesh.cu
    src/RemeshingHelper.cu
    src/PhysicsHelper.cu
    src/CollisionHelper.cu
)
foreach(debug_source IN LISTS CUDA_DEVICE_DEBUG_SOURCES)
    if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${debug_source}" OR
        NOT debug_source MATCHES "\\.cu$")
        message(FATAL_ERROR "GPU debug source must be an existing .cu file: ${debug_source}")
    endif()
    set_property(SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/${debug_source}"
        APPEND PROPERTY COMPILE_OPTIONS "$<$<CONFIG:Debug>:-G>")
endforeach()

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
if (POLICY CMP0146)
    cmake_policy(SET CMP0146 OLD)
endif()
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
