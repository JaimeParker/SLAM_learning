//
// Created by hazyparker on 2021/11/27.
//

#ifndef NONLINEAR_OPTIMIZATION_LMMETHOD2_H
#define NONLINEAR_OPTIMIZATION_LMMETHOD2_H
#include <cstdio>
#include <vector>
#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;

void LM(double(*Func)(const Mat &input, const Mat params), // function pointer
        const Mat &inputs, const Mat &outputs, Mat& params);

double Deriv(double(*Func)(const Mat &input, const Mat params), // function pointer
             const Mat &input, const Mat params, int n);

// The user defines their function here
double Func(const Mat &input, const Mat params);

void LMSolve2();

#endif //NONLINEAR_OPTIMIZATION_LMMETHOD2_H
