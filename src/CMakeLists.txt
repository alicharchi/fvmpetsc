cmake_minimum_required(VERSION 2.8...3.13)
project(petscTest C)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake-modules")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_VERBOSE_MAKEFILE OFF)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
set(CMAKE_C_FLAGS_DEBUG "-g")
set(CMAKE_C_FLAGS_RELEASE "-O3")

# Find pkg-config tool
find_package(PkgConfig REQUIRED)

find_package(MPI REQUIRED)

# Use pkg-config to find PETSc
pkg_check_modules(PETSC REQUIRED PETSc)

# Include PETSc headers
include_directories(${PETSC_INCLUDE_DIRS})
link_directories(${PETSC_LIBRARY_DIRS})

add_executable(twoD twoD.c)
target_link_libraries(twoD m petsc MPI::MPI_C)

add_executable(oneD oneD.c)
target_link_libraries(oneD m petsc MPI::MPI_C)