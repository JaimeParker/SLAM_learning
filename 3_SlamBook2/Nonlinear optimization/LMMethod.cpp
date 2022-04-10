//
// Created by hazyparker on 2021/11/26.
//

/**
 * Levenberg-Marquardt Algorithm
 * using C++, programme based on GaussNewton.cpp, applying the same model
 * reference:
 * Methods for Non-Linear Least Square Problem by DTU(http://www2.imm.dtu.dk/pubdb/edoc/imm3215.pdf)
 * SLAM BOOK2(https://github.com/gaoxiang12/slambook2)
 */

#include "LMMethod.h"
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <chrono>

using namespace std;
using namespace Eigen;

void TrustRegionMethod(){
    cout << "Levenberg-Marquardt Algorithm" << endl;
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
    double cost, error, lastCost = 0;
    double tau = 0.001 * 0.001;
    double v = 2.0;
    double GainRatio;
    // get mu_0
    Matrix3d H_0 = Matrix3d::Zero();
    Vector3d J_0 = Vector3d::Zero();
    double x0 = x_data[0];
    J_0[0] = -x0 * x0 * exp(ae * x0 * x0 + be * x0 + ce);
    J_0[1] = -x0 * exp(ae * x0 * x0 + be * x0 + ce);
    J_0[2] = exp(ae * x0 * x0 + be * x0 + ce);
    H_0 = J_0 * J_0.transpose();
    double mu = max(H_0(0, 0) ,H_0(1, 1));
    mu = tau * max(mu, H_0(2, 2));

    Matrix3d I = Matrix3d::Identity();
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    for (int iter = 0; iter < iteration; iter++){
        // define matrix H, g
        Matrix3d H = Matrix3d::Zero();
        Vector3d g = Vector3d::Zero();
        Vector3d J_sum = Vector3d::Zero();
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
            J_sum = J_sum + J;
            // define cost
            cost =  cost + error * error;
            // update matrix H, g
            H = H + J * J.transpose() * inv_sigma * inv_sigma;
            g = g - error * J;
        }

        // solve equation H * dx = g
        H = H + mu * I;
        Vector3d dx = H.ldlt().solve(g);
        if (dx[0] == 0){
            cout << "fail to solve matrix H, it can't be inverse" << endl;
            break;
        }
        // update estimated values
        ae = ae + dx[0];
        be = be + dx[1];
        ce = ce + dx[2];
        // solve Gain Ratio
        GainRatio = (cost - lastCost) / (0.5 * J_sum.transpose() * (mu * J_sum - g));
        lastCost = cost;
        if (GainRatio > 0){
            v = 2;
            mu = mu * max(1.0 / 3.0, (1 - pow((2 * GainRatio - 1), 3)));

        }if (GainRatio <= 0){
            mu = mu * v;
            v = 2 * v;
        }

        // set stopping criteria
        if (iter >= iteration){
            cout << "MAX iteration!" << endl;
            break;
        }if (dx.lpNorm<1>() <= 1e-8){
            cout << "dx Norm<1>: " << dx.lpNorm<1>() << endl;
            cout << "update value(dx) is small enough!" << endl;
            break;
        }if (g.lpNorm<Infinity>() <= 1e-8){
            cout << "g Norm<Infinity>: " << g.lpNorm<Infinity>() << endl;
            cout << "g is small enough!" << endl;
            break;
        }

        cout << "iteration:" << iter << endl;
        cout << "total cost: " << cost << ", \t\tupdate: " << dx.transpose() <<
             "\t\testimated params: " << ae << "," << be << "," << ce << endl;
        cout << "Gain Ratio: " << GainRatio << ", \t\tmu: " << mu << ", \t\tv: " << v << endl;
        cout << endl;
    }

    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "solve time cost = " << time_used.count() << " seconds. " << endl;

    cout << "estimated abc = " << ae << ", " << be << ", " << ce << endl;
    cout << "real abc = " << 1 << ", " << 2 << ", " << 1 << endl;
}