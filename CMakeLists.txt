cmake_minimum_required (VERSION 2.6)
# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()

if (NOT DEFINED CMAKE_MACOSX_RPATH)
   set(CMAKE_MACOSX_RPATH 0)
endif()

# the project name
project(NNAGD)

# activating some global properties for the project
set(NNAGD_VERSION 1.0.0)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(PkgConfig REQUIRED)
pkg_search_module(YAML REQUIRED yaml-cpp)
pkg_search_module(EIGEN3 REQUIRED eigen3)
pkg_search_module(CERES REQUIRED ceres-solver)
pkg_search_module(GLOG REQUIRED libglog)
pkg_search_module(GFLAGS REQUIRED gflags)

include_directories(${PROJECT_SOURCE_DIR}/inc)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${YAML_CFLAGS} ${EIGEN3_CFLAGS} ${CERES_CFLAGS} ${GLOG_CFLAGS} ${GFLAGS_CFLAGS} -std=c++11 -Wall -fPIC")

# build target library
add_subdirectory(src)
add_subdirectory(tests)
