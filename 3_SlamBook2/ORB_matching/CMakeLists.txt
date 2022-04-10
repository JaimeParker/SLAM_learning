cmake_minimum_required(VERSION 3.20)
project(ORB_matching)

set(CMAKE_CXX_STANDARD 14)

# add OpenCV
FIND_PACKAGE(OpenCV REQUIRED)
INCLUDE_DIRECTORIES({OpenCV_INCLUDE_DIRS})

add_executable(ORB_matching main.cpp ORB_cv.cpp ORB_cv.h)

# link
TARGET_LINK_LIBRARIES(ORB_matching ${OpenCV_LIBS})
