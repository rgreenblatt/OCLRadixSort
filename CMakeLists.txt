cmake_minimum_required(VERSION 3.8)
project(OCLRadixSort)

set(CMAKE_CXX_STANDARD 17)

set(OCL_ROOT $ENV{OCL_ROOT} )

add_definitions(-Wall -Werror -Wno-deprecated-declarations -Wno-ignored-attributes -DBOOST_COMPUTE_DEBUG_KERNEL_COMPILATION -DBOOST_COMPUTE_USE_CPP11 -g)

include_directories( ${OCL_ROOT}/include )
link_directories   ( ${OCL_ROOT}/lib/x86_64 )

set(SOURCE_FILES include/OCLRadixSort.hpp)

set(TARGET_NAME OCLRadixSort)
# trivial example
add_executable(${TARGET_NAME}-trivial ${SOURCE_FILES} example/Trivial.cpp)
target_link_libraries(${TARGET_NAME}-trivial OpenCL)
