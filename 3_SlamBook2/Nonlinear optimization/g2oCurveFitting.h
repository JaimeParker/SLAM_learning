//
// Created by hazyparker on 2021/11/27.
//

#ifndef NONLINEAR_OPTIMIZATION_G2OCURVEFITTING_H
#define NONLINEAR_OPTIMIZATION_G2OCURVEFITTING_H

#include <iostream>
#include <g2o/core/g2o_core_api.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <cmath>
#include <chrono>

using namespace std;

void G2OSolve();


#endif //NONLINEAR_OPTIMIZATION_G2OCURVEFITTING_H
