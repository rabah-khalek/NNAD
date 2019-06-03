option(SHARED "Build shared-libray instead of static-libray" ON)
if(SHARED)
  add_library(NNAGD SHARED)
else(SHARED)
  add_library(NNAGD STATIC)
endif(SHARED)

target_link_libraries(NNAGD ${YAML_LDFLAGS} ${EIGEN3_LDFLAGS} ${CERES_LDFLAGS} ${GLOG_LDFLAGS} ${GFLAGS_LDFLAGS})
