//
// Created by JaimeParker on 2021/9/5.
//

#include "myEdge.h"
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

//int lowThreshold = 0; // 怀疑是定义在class中的这个数不会被createTrackbar修改

MyCanny::MyCanny(const Mat &img) {
    // constructor
    /**
     * 注意如果要对类的数据进行赋值操作，一般是 Class class("parameters")这样的语句；
     * 这就是建立在已经书写构造函数的前提下；
     * 如果需要赋值多个数据，建议写在构造函数的参数括号内，进行赋值
     * 构造函数可以有多个，建议写一个构造函数以提供本class中数据的默认值，这样如果用户没有输入也可以运行
     */
    src = img;
}

void MyCanny::canny_process() {
    cout << "canny_process is called..." << endl;
    if (src.empty()){
        cout << "Failed to load src..." << endl;
        exit(100);
    }
    dst.create(src.size(), src.type());
    cvtColor(src, src_gray, COLOR_BGR2GRAY);

    namedWindow(window_name, WINDOW_AUTOSIZE); // 这一段代码必须有，是createTrackbar必须的
    createTrackbar("Min Threshold:",
                   window_name,
                   &lowThreshold,
                   max_lowThreshold,
                   canny_threshold,
                   canny_pointer);
    canny_threshold(0, canny_pointer); // callback function 回调函数
    // FIXME: 回调时，lowThreshold这个参数怎么传给Canny_threshold? 通过`pos`
    waitKey(0);
}

void MyCanny::canny_threshold(int pos, void *userdata) {
    // 可以把自己数据封装到结构体、或者类，用usrdata传进来。另外一个参数`pos`是当前trackbar的数值
    // 实例化一个对象，然后通过对象指针访问非静态成员数据
    auto * myCanny = (MyCanny *) userdata;

    // 这几行代码测试是否被回调了
    cout << "canny_threshold is called..." << endl;
    cout << "pos=" << pos << endl;
    cout << "lowThreshold=" <<myCanny->lowThreshold << endl;
    // 通过一个check_int来查看数据是否送入
    if (myCanny->check_int == 1) cout << "Error! data access FAILED..." << endl;
    else cout << "data access SUCCESSFULLY..." << endl;
    cout << "max_lowThreshold=" << myCanny->max_lowThreshold << endl;

    // 获取类的数据，简写
    Mat src = myCanny->src;
    Mat dst = myCanny->dst;
    Mat src_gray = myCanny->src_gray;
    Mat detected_edges = myCanny->detected_edges;
    int ratio = myCanny->ratio;
    int kernel_size = myCanny->kernel_size;
    // 更新lowThreshold
    myCanny->lowThreshold = pos;

    // 此时这些参数都是用实例化的对象，通过类传过来的
    blur(src_gray, detected_edges, Size(3, 3));
    cout << "blur finished..." << endl;
    Canny(detected_edges,
          detected_edges,
          myCanny->lowThreshold,
          myCanny->lowThreshold * ratio,
          kernel_size);
    cout << "canny finished..." << endl;
    dst = Scalar::all(0);
    src.copyTo(dst, detected_edges);
    imshow(myCanny->window_name, dst);

    cout << endl;
}

MyCanny::MyCanny() {
    src = imread("../../images/cloud.png");
}

int MyCanny::set_check_int(int i) {
    check_int = i;
    return check_int;
}

void function2_1(){
    // 实现Canny边缘检测，用类的方法
    cout << "Canny Edge Detector" << endl;
    Mat img = imread("../../images/learn_slam.jpg");
    MyCanny mCanny(img);
    mCanny.set_check_int(22);
    mCanny.canny_process();
}