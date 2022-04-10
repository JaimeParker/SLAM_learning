//
// Created by JaimeParker on 2021/8/8.
//

#ifndef INC_1_OPENCV_TEST1_H
#define INC_1_OPENCV_TEST1_H

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;


class test1 {

};

class basicDrawing{
    /** 关于本类和方法的使用
     * 基本的绘图操作，主要源自OpenCV document的原码
     * https://docs.opencv.org/4.5.2/d3/d96/tutorial_basic_geometric_drawing.html
     * 话说自己用的东西还用写这么多注释吗？
     */
private:
    Mat img1;

public:
    explicit basicDrawing(Mat img1);
    Mat get_image();

    void testLine(Point start, Point end);

    static void mLine(Mat img, Point start, Point end);
    void mEllipse(Mat img, double angle);
    static void mPolygon(Mat img);
    static void mFilledCircle(Mat img, Point center);
};

class MyCV{
    /** 关于本类和方法的使用
     * 相当于提供了一个图片侵蚀和扩张（形态学处理）的接口
     * 可以通过.compose_erode和.compose_dilate来直接调用这两种方法
     */
private:
    Mat src;
    Mat dst;
    Mat show_img;

    int erosion_elem = 0;
    int erosion_size = 0;
    int dilation_elem = 0;
    int dilation_size = 0;
    int const max_elem = 2;
    int const max_kernel_size = 21;

    int morph_elem = 0; // 决定element矩阵的样式
    int morph_size = 0;
    int morph_operator = 0; //morph 的操作数
    int const max_operator = 4;
    const char * window_name_morph = "morphology operation";



public:
    MyCV();
    explicit MyCV(Mat& img);
    void set_src(Mat &img);
    void show();
    void compose_erode();
    void compose_dilate();
    static void erode_callback(int pos, void* userdata);
    static void dilate_callback(int pos, void* userdata);
    static void dilate_elem_callback(int pos, void* userdata);

    MyCV * cv_pointer = this;

    int set_erosion_elem(int i);
    int set_dilation_elem(int i);
    /**
     * 可以在 set_erosion_elem, set_dilation_elem这种函数里面定义elem
     * 也可以两个trackbar,实现在window里面有两个可以调节的trackbar
     * 现在看来，只要按顺序回调就可以了，详见compose_dilate()函数
     */
    void compose_gaussian_blur(); // 对一张图片执行高斯模糊处理
    void morphology_operation(); // 形态学操作的主函数
    void set_morphology_operator(int i);
    /**
     * send morph_operator to variable 'morph_operator'
     * morph_operator = 0, opening
     * morph_operator = 1, closing
     * morph_operator = 2, gradient
     * morph_operator = 3, top hat
     * morph_operator = 4, black hat
     */

    static void morphology_elem_callback(int pos, void * userdata);
    /**
     * @param pos, sent morph_elem from createTrackbar to this callback function
     * @param userdata, linked with userdata, namely the class-objective, myCV
     * morph_elem = 0, Rect
     * morph_elem = 1, Cross
     * morph_elem = 2, Ellipse
     */

    static void morphology_size_callback(int pos, void * userdata); // morphology_operation函数的第二个callback

    void extract_lines();
    /**
     * input image, whatever RGB or gray scale
     * using cvtColor, change input image to gray scale
     * using adaptiveThreshold, change input image to binary image
     * define structureElement
     * open in morphology operation, extracting horizontal or vertical lines
     */

    void erase_line(int blockSize);
    /**
     * to erase lines in an image
     * the outcome highly depends on blockSize and
     * structureElement size
     */

    void image_pyramid(int zoom_scale, int type);
    /**
     * Image Pyramid, https://docs.opencv.org/4.5.2/d4/d1f/tutorial_pyramids.html
     * function? keep the main feature of a picture, similar to zoom in/out, but different
     * what is so called feature, keep focus on the principle
     * @param zoom_scale, the scale^2 of zoom in or out, such as 4, 9, 16, etc
     * @param type, type = 0 means zooming in, type = 1 means zooming out
     */

    void threshold_operation() const;
    /**
     * basic threshold operation
     * using a threshold to divide images, binary segmentation(2 type)
     * class: threshold binary,
     * class: threshold binary inverse, converse to threshold binary
     * class: truncate截断，大于则等于阈值，小于则不标
     * class: threshold to zero, 大于阈值不变，小于阈值取零
     * class: threshold to zero inverse,
     * THERSG_OTSU: choose optimal threshold value
     */

    static void threshold_operation_callback(int pos, void * userdata);

    /**
     * create your own linear filters, https://docs.opencv.org/4.5.2/d4/dbd/tutorial_filter_2d.html
     * using filter2D()
     * @param type, ROBERT_X = 0, ROBERT_Y = 1, SOBEL_X = 2, SOBEL_Y = 3, LAPLACE = 4
     */
    void self_defined_filter(int type);

    // FIXME：这玩意怎么用？
    enum MyKernelTypes{
        ROBERT_X = 0,
        ROBERT_Y = 1,
        SOBEL_X = 2,
        SOBEL_Y = 3,
        LAPLACE = 4
    };

    /**
     * add edges of an image
     */
    void add_edges();

    /**
     * 边缘，像素值发生跃迁的地方，对图像取一阶导数，得到其图像像素变化率，根据导数判断边缘
     * sobel算子，用于计算图像灰度的近似梯度，得到的是图像在xy方向的梯度图像
     * 又被称为一阶微分算子，求导算子
     * 最终梯度为了减少cpu开销，改为xy的绝对值求和
     *
     */
    void sobel_operation();

    /**
     * 求二阶导，判断边缘；处理流程如下
     * 高斯模糊，去噪声 GaussianBlur()
     * 转灰度 cvtColor()
     * 拉普拉斯计算 Laplacian()
     * 取绝对值 convertTo()
     * 显示结果
     */
    void laplace_operation();

    void canny_process();

    void canny_threshold_callback();

    void extract_lines_pro();

    void extract_circle();

    void compose_remapping();



    // 现在大致的学习流程是：
    // 一方面在openCV的算法原理，函数接口和使用上学习
    // 一方面在openCV例子的实现中，提高自己C++编程的能力和手感
    // 形成自己的代码风格
};



void function1_0();
void function1_1();
void function1_2();
void function1_3();
void function1_4();
void function1_5();
void function1_6();
void function1_7();
void function1_8();
void function1_9();
void function1_10();
void function1_11();
void function1_12();
void function1_13();
void function1_14();
void function1_15();
void function1_16();

#endif //INC_1_OPENCV_TEST1_H
