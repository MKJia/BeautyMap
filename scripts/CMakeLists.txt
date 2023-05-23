cmake_minimum_required(VERSION 3.14)
project(dufomap_benchmark
	VERSION 1.0.0
	DESCRIPTION "DUFOMap benchmark tools, including data processing, evaluation, and visualization."
	LANGUAGES CXX
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
include_directories(include)
set (CMAKE_CXX_FLAGS "-lstdc++fs -std=c++17")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
find_package(Glog REQUIRED)
find_package(gflags REQUIRED)
set(LOG_LIB glog::glog gflags)
find_package(PCL REQUIRED)

function(create_executable target_name source_file)
    add_executable(${target_name} ${source_file})
    target_link_libraries(${target_name} ${PCL_LIBRARIES} ${LOG_LIB})
endfunction()

create_executable(export_eval_pcd export_eval_pcd.cpp)
