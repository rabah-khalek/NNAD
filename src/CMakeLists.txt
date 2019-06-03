set(src_source
  AnalyticCostFunction.cc
  AutoDiffCostFunction.cc
  NumericCostFunction.cc
  Utilities.cc
  )

add_library(src OBJECT ${src_source})

option(SHARED "Build shared-libray instead of static-libray" ON)
if(SHARED)
  add_library(NNAGD SHARED $<TARGET_OBJECTS:src>)
else(SHARED)
  add_library(NNAGD STATIC $<TARGET_OBJECTS:src>)
endif(SHARED)

target_link_libraries(NNAGD ${YAML_LDFLAGS} ${EIGEN3_LDFLAGS} ${CERES_LDFLAGS} ${GLOG_LDFLAGS} ${GFLAGS_LDFLAGS})
