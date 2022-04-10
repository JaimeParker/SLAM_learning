//
// Created by JaimeParker on 2021/9/5.
//

#ifndef INC_1_OPENCV_MYEDGE_H
#define INC_1_OPENCV_MYEDGE_H


#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

class MyCanny{
private:
    Mat src, src_gray;
    Mat dst, detected_edges; // input and output matrix
    int lowThreshold = 0;
    const int max_lowThreshold = 100;
    const int ratio = 3;
    const int kernel_size = 3;
    const char* window_name = "Edge Map"; // some parameters
    int check_int = 1;
    MyCanny * canny_pointer = this;

public:
    explicit MyCanny(const Mat &img); // 构造函数，用于对类的对象赋值，由于数据是private的，只能通过此种方式赋值
    MyCanny(); // 构造函数，用于初始化
    void canny_process(); // 用于进行主要的CannyEdge处理过程
    static void canny_threshold(int pos, void* userdata);
    int set_check_int(int i); // 一个简易的数据读取测试
};

void function2_1();


#endif //INC_1_OPENCV_MYEDGE_H
