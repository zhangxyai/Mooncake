add_library(cncl_transport OBJECT cncl_transport.cpp)
target_include_directories(cncl_transport
                           PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../../../include)
target_link_libraries(cncl_transport PRIVATE cncl cnrt cndrv JsonCpp::JsonCpp
                                             glog::glog pthread)
target_compile_features(cncl_transport PRIVATE cxx_std_17)
