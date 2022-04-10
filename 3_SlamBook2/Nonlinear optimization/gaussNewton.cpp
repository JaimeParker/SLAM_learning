//
// Created by hazyparker on 2021/11/25.
//

/**
 * Describe the model, y=exp(ax^2+bx+c)+w, which is a nonlinear curve
 * a,b,c are parameters of the curve, w being noise, who meets Gaussian Distribution with means equaling to 0
 * step1: compose value of x and y based on that model
 * step2: add noise(Gaussian Distribution, white noise) to x and y
 * step3: use Gauss-Newton method to solve this model with noise
 * H(x)\Delta x = g, J(x) J(x)^T = -J(x) f(x)
 */

#include "gaussNewton.h"
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <chrono>

using namespace std;
using namespace Eigen;


void GaussNewtonSolve(){
    cout << "Gaussian-Newton Method Solving Nonlinear Optimization" << endl;

    // data preparation
    double ar = 1.0, br = 2.0, cr = 1.0; // real parameters
    double ae = 2.0, be = -1.0, ce = 5.0; // estimated parameters
    int N = 100; // data numbers
    double w_sigma = 1.0; // sigma of noise
    double inv_sigma = 1.0 / w_sigma;
    cv::RNG randomNumber; // make random numbers
    vector<double> x_data, y_data; // set data container of x and y

    // compose values
    for (int i = 0; i < N; i++){
        double x = i / 100.0, y; // set x_range (0, 1)
        y = exp(ar * x * x + br * x + cr) + randomNumber.gaussian(w_sigma * w_sigma);
        x_data.push_back(x);
        y_data.push_back(y);
    }

    // Gauss-Newton method, starting iteration
    int iteration = 100;
    double cost = 0, error = 0, lastCost = 0;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    for (int iter = 0; iter < iteration; iter++){
        // define matrix H, g
        Matrix3d H = Matrix3d::Zero();
        Vector3d g = Vector3d::Zero();
        cost = 0; // set cost zero for a new iteration

        for (int i = 0; i < N; i++){
            // get error, f(x)
            double xi, yi;
            yi = y_data[i];
            xi = x_data[i];
            error = yi - exp(ae * xi * xi + be * xi + ce);

            // get matrix J
            Vector3d J = Vector3d::Zero();
            J[0] = -xi * xi * exp(ae * xi * xi + be * xi +ce);
            J[1] = -xi * exp(ae * xi * xi + be * xi +ce);
            J[2] = -exp(ae * xi * xi + be * xi +ce);

            // define cost
            cost =  cost + error * error;

            // update matrix H, g
            H = H + J * J.transpose();
            g = g - error * J;
        }

        // solve equation H * dx = g
        Vector3d dx = H.ldlt().solve(g);
        if (dx[0] == 0){
            cout << "fail to solve matrix H, it can't be inverse" << endl;
            break;
        }

        // set break ctor
        if (iter > 0 && cost >= lastCost){
            cout << "cost: " << cost << ">= last cost: " << lastCost << ", break." << endl;
            break;
        }

        // update estimated values
        ae = ae + dx[0];
        be = be + dx[1];
        ce = ce + dx[2];

        // redefine cost
        lastCost = cost;

        cout << "iteration:" << iter << endl;
        cout << "total cost: " << cost << ", \t\tupdate: " << dx.transpose() <<
             "\t\testimated params: " << ae << "," << be << "," << ce << endl;

    }

    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "solve time cost = " << time_used.count() << " seconds. " << endl;

    cout << "estimated abc = " << ae << ", " << be << ", " << ce << endl;
}


