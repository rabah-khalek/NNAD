set(src_source
  AnalyticCostFunction.cc
  AutoDiffCostFunction.cc
  NumericCostFunction.cc
  )

option(SHARED "Build shared-libray instead of static-libray" ON)
if(SHARED)
  add_library(src SHARED ${src_source})
else(SHARED)
  add_library(src STATIC ${src_source})
endif(SHARED)

target_link_libraries(src ${YAML_LDFLAGS} ${EIGEN3_LDFLAGS} ${CERES_LDFLAGS} ${GLOG_LDFLAGS} ${GFLAGS_LDFLAGS})
