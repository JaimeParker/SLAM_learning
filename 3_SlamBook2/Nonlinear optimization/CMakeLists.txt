cmake_minimum_required(VERSION 3.20)
project(Nonlinear_optimization)

set(CMAKE_CXX_STANDARD 14)

# add OpenCV
find_package(OpenCV REQUIRED)
include_directories({OpenCV_INCLUDE_DIRS})

# add Eigen
include_directories("/usr/local/include/eigen3")

# Ceres
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

# g2o
list(APPEND CMAKE_MODULE_PATH /home/hazyparker/3rdParty/g2o/cmake_modules)
find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIRS})

add_executable(Nonlinear_optimization main.cpp
        gaussNewton.cpp gaussNewton.h
        LMMethod.cpp LMMethod.h
        LMMethod2.cpp LMMethod2.h
        ceresCurveFitting.cpp ceresCurveFitting.h g2oCurveFitting.cpp g2oCurveFitting.h)

# target link
TARGET_LINK_LIBRARIES(Nonlinear_optimization ${OpenCV_LIBS} ${CERES_LIBRARIES}  ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY})

# please make sure you are using Linux System and installed with OpenCV and Eigen3
# Suggestion: only add codes and CMakeList to VCS(Git)
# Eigen3 version: not a template one, please be advised
