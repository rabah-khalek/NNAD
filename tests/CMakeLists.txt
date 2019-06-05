cmake_minimum_required (VERSION 2.6)
project (Tutorial)

if (NOT DEFINED CMAKE_MACOSX_RPATH)
   set(CMAKE_MACOSX_RPATH 0)
endif()

add_compile_options(-std=c++11 -Wall -fPIC)

# Activating some global properties for the project
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include relevant packages
find_package(PkgConfig REQUIRED)
pkg_search_module(YAML REQUIRED yaml-cpp)
pkg_search_module(EIGEN3 REQUIRED eigen3)
pkg_search_module(CERES REQUIRED ceres-solver)
pkg_search_module(GLOG REQUIRED libglog)
pkg_search_module(GFLAGS REQUIRED gflags)
find_program(NNAD_CONFIG nnad-config REQUIRED)
if (NNAD_CONFIG)
  exec_program(${NNAD_CONFIG}
    ARGS --cflags
    OUTPUT_VARIABLE NNAD_CFLAGS
  )
endif(NNAD_CONFIG)

include_directories(${PROJECT_SOURCE_DIR}/inc)
include_directories(${PROJECT_SOURCE_DIR}/src)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${NNAD_CFLAGS} ${YAML_CFLAGS} ${EIGEN3_CFLAGS} ${CERES_CFLAGS} ${GLOG_CFLAGS} ${GFLAGS_CFLAGS}")

add_subdirectory(src) 
add_subdirectory(run)
