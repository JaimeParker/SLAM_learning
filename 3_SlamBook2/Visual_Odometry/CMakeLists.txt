cmake_minimum_required(VERSION 3.21)
project(Visual_Odometry)

set(CMAKE_CXX_STANDARD 14)

# find package
find_package(OpenCV REQUIRED)

add_executable(Visual_Odometry main.cpp)
add_executable(orb_cv orb_cv.cpp orb_cv.h)

# target link
target_link_libraries(orb_cv ${OpenCV_LIBS})