cmake_minimum_required(VERSION 3.20)
project(xexprep CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
# XenonRecomp's tree (extern/XenonRecomp) provides XenonUtils and its loader.
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../extern/XenonRecomp ${CMAKE_BINARY_DIR}/XenonRecomp EXCLUDE_FROM_ALL)
add_executable(xexprep main.cpp)
target_link_libraries(xexprep PRIVATE XenonUtils)
