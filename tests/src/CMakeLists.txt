set(src_source
  AnalyticCostFunction.cc
  AutoDiffCostFunction.cc
  NumericCostFunction.cc
  )

add_library(src OBJECT ${src_source})

option(SHARED "Build shared-libray instead of static-libray" ON)
if(SHARED)
  add_library(Tutorial SHARED $<TARGET_OBJECTS:src>)
else(SHARED)
  add_library(Tutorial STATIC $<TARGET_OBJECTS:src>)
endif(SHARED)

if (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  target_link_libraries(Tutorial ${YAML_LDFLAGS} ${EIGEN3_LDFLAGS} ${CERES_LDFLAGS} ${GLOG_LDFLAGS} ${GFLAGS_LDFLAGS} -lpthread -fopenmp) 
else()
  target_link_libraries(Tutorial ${YAML_LDFLAGS} ${EIGEN3_LDFLAGS} ${CERES_LDFLAGS} ${GLOG_LDFLAGS} ${GFLAGS_LDFLAGS})
endif() 
