include_directories(${PROJECT_SOURCE_DIR}/tests/inc)
add_subdirectory(src)

# build exacutable
add_executable(main main.cc)
target_link_libraries(main NNAGD CERES)
