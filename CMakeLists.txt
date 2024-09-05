cmake_minimum_required (VERSION 3.5)
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

# The project name
project(NNAD CXX)

set(CMAKE_CXX_STANDARD 17)

# Activating some global properties for the project
set(NNAD_VERSION 2.0.0)

# Configuration script
set(prefix ${CMAKE_INSTALL_PREFIX})
set(exec_prefix "${prefix}")
set(includedir "${prefix}/include")
set(libdir "${prefix}/lib")
configure_file("${PROJECT_SOURCE_DIR}/bin/nnad-config.in" "${PROJECT_BINARY_DIR}/bin/nnad-config")

include_directories(${PROJECT_SOURCE_DIR}/inc)

option(BUILD_TESTS "Build tests code" ON)
if(BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif(BUILD_TESTS)

# Install relevant file
install(FILES ${PROJECT_BINARY_DIR}/bin/nnad-config
	   DESTINATION bin
	   PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/inc/NNAD DESTINATION include)
