cmake_minimum_required(VERSION 3.20)
project(1_OpenCV)

set(CMAKE_CXX_STANDARD 14)

add_executable(1_OpenCV main.cpp mCode/test1.cpp mCode/test1.h mCode/myEdge.cpp mCode/myEdge.h)

set (OpenCV_DIR "D:\\ProgramingSW\\openCV\\opencv_sub\\opencv\\mingw-build")
FIND_PACKAGE(OpenCV REQUIRED)
IF (OpenCV_FOUND)
    INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
    TARGET_LINK_LIBRARIES(1_OpenCV ${OpenCV_LIBS})
ELSE (OpenCV_FOUND)
    MESSAGE(FATAL_ERROR "OpenCV library not found")
ENDIF (OpenCV_FOUND)
