cmake_minimum_required(VERSION 2.8...3.13)
project (petscTest)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake-modules")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_VERBOSE_MAKEFILE OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++0x -pthread")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_CXX_COMPILER "mpic++")

# Find pkg-config tool
find_package(PkgConfig REQUIRED)

# Use pkg-config to find PETSc
pkg_check_modules(PETSC REQUIRED PETSc)

# Include PETSc headers
include_directories(${PETSC_INCLUDE_DIRS})
link_directories(${PETSC_LIBRARY_DIRS})

add_executable(twoD twoD.cpp)
target_link_libraries(twoD m petsc)

add_executable(oneD oneD.cpp)
target_link_libraries(oneD m petsc)
