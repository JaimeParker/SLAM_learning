cmake_minimum_required(VERSION 3.21)
project(loop_closure)

set(CMAKE_CXX_STANDARD 14)

# find package: opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# find package: DBoW3
find_package(DBoW3 REQUIRED)
include_directories(${DBoW3_INCLUDE_DIRS})

# add executable and link libraries
add_executable(loop_closure main.cpp)
add_executable(similarity_test similarity_test.cpp similarity_test.h)
target_link_libraries(similarity_test ${OpenCV_LIBS} ${DBoW3_LIBS})
target_link_libraries(loop_closure ${OpenCV_LIBS} ${DBoW3_LIBS})