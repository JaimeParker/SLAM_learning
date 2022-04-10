//
// Created by JaimeParker on 2021/8/8.
//

#include "test1.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <utility>


using namespace std;
using namespace cv;

void function1_0(){
    // Mat复制操作
    Mat A;
    A = imread("D:/Downloads/Browser/cloud.png", IMREAD_COLOR);
    Mat B(A);
    Mat D(A, Rect(30, 30, 100, 100));
    Mat E = A(Range::all(), Range(1, 20));
    Mat F;
    F.create(200, 200, CV_8UC(2));
    Mat M(200, 200, CV_8UC3, Scalar(0, 0, 225));
    int sz[3] = {2, 2, 2};
    Mat L(3, sz, CV_8UC(1), Scalar::all(0));

    imshow("A", A);
    imshow("B", B);
    imshow("D", D);
    imshow("E", E);
    imshow("F", F);
    imshow("M", M);

    // LUT
    uchar lutData[256];
    for (int i = 0; i < 256; ++i){
        lutData[i] = 255 - i;
    }
    Mat lutTable(1,256, CV_8U);
    uchar *p = lutTable.ptr();
    for (int i = 0; i < 256; ++i){
        p[i] = lutData[i];
    }
    Mat G;
    LUT(A, lutTable, G);
    imshow("G, after lut", G);

    Mat myPhoto = imread("D:/DCIMed/picked/DSC00510-1.jpg");
    Mat myPhotoEd;
    LUT(myPhoto, lutTable, myPhotoEd);
    namedWindow("raw", WINDOW_NORMAL);
    namedWindow("edited", WINDOW_NORMAL);
    imshow("raw", myPhoto);
    imshow("edited", myPhotoEd);
    imwrite("D:/Personal Files/project files/PI-LAB/learn_slam/images/raw1.jpg", myPhoto);
    imwrite("D:/Personal Files/project files/PI-LAB/learn_slam/images/edited1.jpg", myPhotoEd);
    waitKey(0);
}

void function1_1(){
    // 加载图片并转换为灰度图
    cout << "Work1-1" << endl;
    string srcPath = "D:/Downloads/Browser/cloud.png";
    string dstPath = "D:/Personal Files/project files/PI-LAB/learn_slam/images/cloud_gray.png";
    // image read path (R"(D:\Downloads\Browser\cloud.png)");
    // 上面这种写法是ClangTidy建议的，Raw地址的写法，实际写到双引号里面是\\的形式
    // 个人感觉地址单独列出的形式，封装性更好，也更清楚
    Mat src = imread(srcPath);
    Mat dst;
    if (src.empty()){
        cout << "error! image load failed" << endl;
    }else{
        cout << "loading image..." << endl;
    }
    cvtColor(src, dst, COLOR_BGR2GRAY); // 注意这里是OpenCV4.2.0的参数，之前的是CV_RGB2GRAY
    imshow("input", src);
    imshow("output", dst);
    imwrite(dstPath, dst);
    waitKey(0);
}

void function1_2(){
    // 矩阵的掩膜操作
    cout << "Mat mask/kernel" << endl;
    Mat src, dst;
    string srcPath = "D:/Personal Files/project files/PI-LAB/learn_slam/images/learn_slam.jpg";
    src = imread(srcPath);
    if (src.empty()){
        // 也可以用!src.data()
        cout << "couldn't load the image" << endl;
        exit(100);
    }
    //获取图像的长度和宽度，但需要考虑channel
    int cols = src.cols * src.channels();
    int rows = src.rows;
    int srcChannels = src.channels();
    dst.create(src.size(), src.type()); //输出图像初始化
    cout << "cols: " << src.cols << ", rows: " << src.rows << endl;

    // Mat kernel operation
    for (int row = 1; row < rows - 1; row++){
        const uchar* previous = src.ptr(row - 1);
        const uchar* current = src.ptr(row);
        const uchar* next = src.ptr(row + 1);
        uchar* output = dst.ptr(row);
        for (int col = srcChannels; col < cols - srcChannels; col++){
            output[col] = saturate_cast<uchar>(5 * current[col] - (current[col - srcChannels] + current[col + srcChannels] +
                    previous[col] + next[col]));
        }
    }

    namedWindow("output", WINDOW_AUTOSIZE);
    imshow("output", dst);
    namedWindow("src", WINDOW_AUTOSIZE);
    imshow("src", src);
    imwrite("D:/Personal Files/project files/PI-LAB/learn_slam/images/learn_slam_edit.jpg", dst);

    // test pixel intensity acquiring
    int x, y;
    cout << "input coordinate x, y:" << endl;
    cin >> x >> y;
    Vec3b intensity = src.at<Vec3b>(y, x);
    uchar blue = intensity.val[0];
    uchar green = intensity.val[1];
    uchar red = intensity.val[2];
    cout << "RGB:" << int(red) << ", " << int(green) << ", " << int(blue) << endl;

    waitKey(0);

    /**
     * openCV中给出了掩膜函数，filter2D
     * Mat kernel = (Mat_<char>(3, 3) << 0, -1, 0,
     *                                  -1, -5, -1,
     *                                   0, -1, 0);
     * filter2D(src, dst, src.depth(), kernel)
     */
}

void function1_3(){
    //混合两张图片，线性融合，交叉溶解
    cout << "add/blend two images" << endl;

    double alpha = 0.5; double beta; double gamma = 0;

    Mat src1, src2, dst;

    src1 = imread("D:/Personal Files/project files/PI-LAB/learn_slam/images/test1-3ed1.jpeg");
    src2 = imread("D:/Personal Files/project files/PI-LAB/learn_slam/images/test1-3ed.jpeg");
    cout << "src1 channels:" << src1.channels() << ", " << "src1 size:" << src1.rows << "*" << src1.cols << endl;
    cout << "src2 channels:" << src2.channels() << ", " << "src2 size:" << src2.rows << "*" << src2.cols << endl;
    if (src1.empty()){
        cout << "can't load that image..." << endl;
        exit(100);
    }

    beta = 1 - alpha;
    addWeighted(src1, alpha, src2, beta, gamma, dst);

    imshow("Linear blend", dst);
    waitKey(0);
    imwrite("D:/Personal Files/project files/PI-LAB/learn_slam/images/test1-3final.jpeg", dst);
}

void function1_4(){
    //对比度和亮度操作
    cout << "Changing contrast and brightness" << endl;
    Mat image = imread("D:/Personal Files/project files/PI-LAB/learn_slam/images/test1-3ed1.jpeg");

    Mat new_image = Mat::zeros( image.size(), image.type() );

    double alpha = 1.0; /* Simple contrast control */
    int beta = 0;       /* Simple brightness control */

    cout << " Basic Linear Transforms " << endl;
    cout << "-------------------------" << endl;
    cout << "* Enter the alpha value [1.0-3.0]: "; cin >> alpha;
    cout << "* Enter the beta value [0-100]: ";    cin >> beta;

    for (int i = 0; i < image.rows; i++){
        for (int j = 0; j < image.cols; j++){
            for (int c = 0; c < image.channels(); c++){
                new_image.at<Vec3b>(i, j)[c] =
                        saturate_cast<uchar>( alpha*image.at<Vec3b>(i, j)[c] + beta );
            }
        }
    }

    imshow("Original Image", image);
    imshow("New Image", new_image);
    waitKey(0);
    imwrite("D:/Personal Files/project files/PI-LAB/learn_slam/images/test1-4.jpeg", new_image);

}

void function1_5(){
    // 离散傅里叶变换
    cout << "DFT: Discrete Fourier Transform" << endl;

    // read image
    Mat I = imread("../../images/cloud.png", IMREAD_GRAYSCALE);
    if (I.empty()){
        cout << "Error! Image load failed..." << endl;
        exit(100);
    }

    // expand the image to optimal size
    Mat padded;
    int m = getOptimalDFTSize( I.rows );
    int n = getOptimalDFTSize( I.cols );
    copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));

    // make place for real and complex values
    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complexI;
    merge(planes, 2, complexI);

    // make discrete transform
    dft(complexI, complexI);

    // trans IM and RE values to magnitude
    split(complexI, planes);
    magnitude(planes[0], planes[1], planes[0]);
    Mat magI = planes[0];

    // transfer to log scale
    magI += Scalar::all(1);
    log(magI, magI);

    // crop and rearrange
    magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));
    int cx = magI.cols/2;
    int cy = magI.rows/2;
    Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
    Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
    Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right
    Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);

    // normalize
    normalize(magI, magI, 0, 1, NORM_MINMAX);

    // reload
    namedWindow("input", WINDOW_AUTOSIZE);
    namedWindow("output", WINDOW_AUTOSIZE);
    imshow("input", I);
    imshow("output", magI);
    imwrite("D:/Personal Files/project files/PI-LAB/learn_slam/images/test1-5.png", magI);
    waitKey(0);
}

void function1_6(){
    // 基本的图像绘制，比如线，椭圆等
    cout << "Work1-6" << endl;
    int w = 400;
    char atom_window[] = "Drawing 1: Atom";
    char rook_window[] = "Drawing 2: Rook";
    Mat atom_image = Mat::zeros( w, w, CV_8UC3 );
    Mat rook_image = Mat::zeros( w, w, CV_8UC3 );
    Mat new_image = Mat::zeros( w, w, CV_8UC3 );

    basicDrawing mImage(new_image);
    mImage.testLine(Point( 0, 15*w/16 ), Point( w, 15*w/16 ));

    mImage.mEllipse( atom_image, 90 );
    mImage.mEllipse( atom_image, 0 );
    mImage.mEllipse( atom_image, 45 );
    mImage.mEllipse( atom_image, -45 );
    basicDrawing::mFilledCircle( atom_image, Point( w/2, w/2) );
    basicDrawing::mPolygon( rook_image );
    rectangle( rook_image,Point( 0, 7*w/8 ),Point( w, w),Scalar( 0, 255, 255 ),FILLED,LINE_8 );
    basicDrawing::mLine( rook_image, Point( 0, 15*w/16 ), Point( w, 15*w/16 ) );
    basicDrawing::mLine( rook_image, Point( w/4, 7*w/8 ), Point( w/4, w ) );
    basicDrawing::mLine( rook_image, Point( w/2, 7*w/8 ), Point( w/2, w ) );
    basicDrawing::mLine( rook_image, Point( 3*w/4, 7*w/8 ), Point( 3*w/4, w ) );

    imshow( atom_window, atom_image );
    moveWindow( atom_window, 0, 200 );
    imshow( rook_window, rook_image );
    moveWindow( rook_window, w, 200 );
    imshow("Image", mImage.get_image());
    waitKey(0);
}

void function1_7(){
    // 线性模糊滤波
    Mat src = imread("D:/Personal Files/project files/PI-LAB/learn_slam/images/test1-3ed1.jpeg");
    Mat dst;
    for ( int i = 1; i < 31; i = i + 2 ){
        GaussianBlur( src, dst, Size( i, i ), 0, 0 );
    }
    imshow("output", dst);
    waitKey(0);
}


basicDrawing::basicDrawing(Mat img1) {
    this->img1=std::move(img1);
}

void basicDrawing::mLine(Mat img, Point start, Point end) {
    int thickness = 2;
    int lineType = LINE_8;
    line(img, start, end, Scalar(255, 255, 255), thickness, lineType);
}

void basicDrawing::mEllipse(Mat img, double angle) {
    int w = 400;
    int thickness = 2;
    int lineType = 8;
    ellipse( img, Point( w/2, w/2 ),Size( w/4, w/16 ),angle,0,360,Scalar( 255, 0, 0 ),thickness,lineType );
}

void basicDrawing::mPolygon(Mat img) {
    int w = 400;
    int lineType = LINE_8;
    Point rook_points[1][20];
    rook_points[0][0]  = Point(    w/4,   7*w/8 );
    rook_points[0][1]  = Point(  3*w/4,   7*w/8 );
    rook_points[0][2]  = Point(  3*w/4,  13*w/16 );
    rook_points[0][3]  = Point( 11*w/16, 13*w/16 );
    rook_points[0][4]  = Point( 19*w/32,  3*w/8 );
    rook_points[0][5]  = Point(  3*w/4,   3*w/8 );
    rook_points[0][6]  = Point(  3*w/4,     w/8 );
    rook_points[0][7]  = Point( 26*w/40,    w/8 );
    rook_points[0][8]  = Point( 26*w/40,    w/4 );
    rook_points[0][9]  = Point( 22*w/40,    w/4 );
    rook_points[0][10] = Point( 22*w/40,    w/8 );
    rook_points[0][11] = Point( 18*w/40,    w/8 );
    rook_points[0][12] = Point( 18*w/40,    w/4 );
    rook_points[0][13] = Point( 14*w/40,    w/4 );
    rook_points[0][14] = Point( 14*w/40,    w/8 );
    rook_points[0][15] = Point(    w/4,     w/8 );
    rook_points[0][16] = Point(    w/4,   3*w/8 );
    rook_points[0][17] = Point( 13*w/32,  3*w/8 );
    rook_points[0][18] = Point(  5*w/16, 13*w/16 );
    rook_points[0][19] = Point(    w/4,  13*w/16 );
    const Point* ppt[1] = { rook_points[0] };
    int npt[] = { 20 };
    fillPoly( img, ppt, npt, 1, Scalar( 255, 255, 255 ), lineType );
}

void basicDrawing::mFilledCircle(Mat img, Point center) {
    int w = 400;
    circle( img, center, w/32, Scalar( 0, 0, 255 ), FILLED,LINE_8 );
}

void basicDrawing::testLine(Point start, Point end){
    int thickness = 2;
    int lineType = LINE_8;
    line(img1, start, end,Scalar(255, 255, 255), thickness, lineType);
}

Mat basicDrawing::get_image()
{
    return this->img1;
}


void function1_8(){
    // 实现erode
    MyCV myCv;
    myCv.set_erosion_elem(2);
    myCv.compose_erode();
}

void function1_9(){
    // 实现dilate
    MyCV myCv;
    myCv.compose_dilate();
}

MyCV::MyCV(Mat& img) {
    src = img;
}

void MyCV::show() {
    namedWindow("output_demo", WINDOW_AUTOSIZE);
    imshow("output_demo", show_img);
    waitKey(0);
}

MyCV::MyCV() {
    // 提供默认值的构造函数
    src = imread("../../images/learn_slam.jpg");
}

void MyCV::compose_erode() {
    namedWindow( "Erosion Demo", WINDOW_AUTOSIZE );
    createTrackbar( "Kernel size:\n 2n +1", "Erosion Demo",
                    &erosion_size, max_kernel_size,
                    erode_callback,
                    cv_pointer);
    erode_callback(0, cv_pointer);
    waitKey(0);
}

void MyCV::compose_dilate() {
    // https://docs.opencv.org/4.5.2/db/df6/tutorial_erosion_dilatation.html
    // 仿照官方代码，实现在一个trackbar里面调节两个参数的函数，注意传参
    namedWindow("Dilation Demo", WINDOW_AUTOSIZE);

    createTrackbar( "Element:\n 0: Rect \n 1: Cross \n 2: Ellipse", "Dilation Demo",
                    &dilation_elem, max_elem,
                    dilate_elem_callback,
                    cv_pointer);
    createTrackbar("Kernel size:\n 2n + 1", "Dilation Demo",
                   &dilation_size, max_kernel_size,
                   dilate_callback,
                   cv_pointer);
    dilate_elem_callback(0, cv_pointer);
    dilate_callback(0, cv_pointer);
    waitKey(0);
}

void MyCV::erode_callback(int pos, void *userdata) {
    cout << "erode_callback is called..." << endl;
    auto * myCV = (MyCV *) userdata;
    int erosion_elem = myCV->erosion_elem;
    Mat erosion_dst = myCV->dst;
    int erosion_size = myCV->erosion_size;
    Mat src = myCV->src;

    int erosion_type = 0;
    if( erosion_elem == 0 ){ erosion_type = MORPH_RECT; }
    else if( erosion_elem == 1 ){ erosion_type = MORPH_CROSS; }
    else if( erosion_elem == 2) { erosion_type = MORPH_ELLIPSE; }

    erosion_size = pos;
    cout << "erosion_size=" << erosion_size << endl;

    Mat element = getStructuringElement( erosion_type,
                                         Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                         Point( erosion_size, erosion_size ) );

    erode( src, erosion_dst, element );
    imshow( "Erosion Demo", erosion_dst );
}

void MyCV::dilate_callback(int pos, void *userdata) {
    cout << "dilate_callback is called..." << endl;
    auto * myCV = (MyCV *) userdata;
    int dilation_elem = myCV->dilation_elem;
    Mat dilation_dst = myCV->dst;
    int dilation_size = myCV->dilation_size;
    Mat src = myCV->src;

    int dilation_type = 0;
    if( dilation_elem == 0 ){ dilation_type = MORPH_RECT; }
    else if( dilation_elem == 1 ){ dilation_type = MORPH_CROSS; }
    else if( dilation_elem == 2) { dilation_type = MORPH_ELLIPSE; }

    dilation_size = pos;
    cout << "dilation_size=" << dilation_size << endl;

    Mat element = getStructuringElement( dilation_type,
                                         Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                         Point( dilation_size, dilation_size ) );
    dilate( src, dilation_dst, element );
    imshow( "Dilation Demo", dilation_dst );
}

int MyCV::set_erosion_elem(int i) {
    erosion_elem = i;
    return erosion_elem;
}

int MyCV::set_dilation_elem(int i) {
    dilation_elem = i;
    return dilation_elem;
}

void MyCV::dilate_elem_callback(int pos, void * userdata) {
    cout << "dilate_elem_callback is called..." << endl;
    auto * myCV = (MyCV *) userdata;
    myCV->dilation_elem = pos;
    cout << "dilation_elem=" << pos << endl;
}

// 大家好我来给大家唱一首，半岛铁盒
// 窗外的麻雀，在电线杆上多嘴
// 不对，这是东风破吗
// 哦，七里香啊
// 希望大家在自己的项目里不要出现这种累了的时候敲废话进去的事情

// 唉，保研政策，一言难尽啊
// 软著买了好几个，代码写了没几行，查也查不出来管也管不了






// 500行纪念！
// >>> 继续努力！

void MyCV::compose_gaussian_blur() {
    for ( int i = 1; i < 31; i = i + 2 ){
        GaussianBlur( src, dst, Size( i, i ), 0, 0 );
    }
    show_img = dst;
}

void MyCV::morphology_operation() {
    namedWindow(window_name_morph, WINDOW_AUTOSIZE);
    createTrackbar( "Element:\n 0: Rect - 1: Cross - 2: Ellipse", window_name_morph,
                    &morph_elem, max_elem,
                    morphology_elem_callback,
                    cv_pointer);
    createTrackbar( "Kernel size:\n 2n +1", window_name_morph,
                    &morph_size, max_kernel_size,
                    morphology_size_callback,
                    cv_pointer);
    morphology_elem_callback(0, cv_pointer);
    morphology_size_callback(0, cv_pointer);

    waitKey(0);
}

void MyCV::set_morphology_operator(int i) {
    morph_operator = i;
}

void MyCV::morphology_size_callback(int pos, void *userdata) {
    auto * myCV = (MyCV *) userdata;
    cout << "morph size is called..." << endl;
    // Since MORPH_X : 2,3,4,5 and 6
    int operation = myCV->morph_operator + 2;
    int morph_elem = myCV->morph_elem;
    int morph_size = myCV->morph_size;
    Mat src = myCV->src;
    Mat dst = myCV->dst;

    Mat element = getStructuringElement( morph_elem,
                                         Size( 2*morph_size + 1, 2*morph_size+1 ),
                                         Point( morph_size, morph_size ) );
    morphologyEx( src, dst, operation, element );
    imshow( myCV->window_name_morph, dst );
}

void MyCV::morphology_elem_callback(int pos, void * userdata) {
    auto * myCV = (MyCV *) userdata;
    cout << "morph elem is called..." << endl;
    myCV->morph_elem = pos;
    cout << "morph_elem=" << myCV->morph_elem << endl;
}

void MyCV::extract_lines() {
    cout << "please input type: horizontal or vertical:";
    string type;
    cin >> type;
    cout << type << endl;
    Mat gray_img;
    cvtColor(src, gray_img, COLOR_BGR2GRAY);
    Mat binary_image;
    adaptiveThreshold(gray_img,
                      binary_image,
                      255,
                      ADAPTIVE_THRESH_MEAN_C,
                      THRESH_BINARY,
                      3, -2); // apply an adaptive threshold to an array
    Mat horizontal_line = getStructuringElement(MORPH_RECT, Size(src.cols / 16, 1), Point(-1, -1));
    Mat vertical_line = getStructuringElement(MORPH_RECT, Size(1, src.cols / 16), Point(-1, -1));

    Mat temp;
    if (type == "horizontal"){
        erode(binary_image, temp, horizontal_line);
        dilate(temp, dst, horizontal_line);
        bitwise_not(dst, dst);
    }if (type == "vertical"){
        erode(binary_image, temp, vertical_line);
        dilate(temp, dst, vertical_line);
        bitwise_not(dst, dst);
    }if (type != "horizontal" && type != "vertical"){
        cout << "what the hell did you input?" << endl;
        exit(100);
    }

    namedWindow("extract lines", WINDOW_AUTOSIZE);
    imshow("extract lines", dst);
    waitKey(0);
}

void MyCV::erase_line(int blockSize) {
    Mat gray_img;
    cvtColor(src, gray_img, COLOR_BGR2GRAY);
    Mat binary_img;
    adaptiveThreshold(gray_img, binary_img,
                      255,
                      ADAPTIVE_THRESH_MEAN_C,
                      THRESH_BINARY,
                      blockSize, -2);

    Mat rect = getStructuringElement(MORPH_RECT, Size(1, 1), Point(-1, -1));

    morphologyEx(binary_img, dst, MORPH_OPEN, rect);
    bitwise_not(dst, dst);
    blur(dst, dst, Size(3, 3));

    namedWindow("delete line", WINDOW_AUTOSIZE);
    imshow("delete line", dst);
    waitKey(0);
}

void MyCV::image_pyramid(int zoom_scale, int type) {
    // 先高斯模糊
    // 再对偶数行进行处理
    // DOG, Difference of Gaussian
    int scale = int(sqrt(zoom_scale));

    if (type == 0){
        pyrUp(src, dst, Size(src.cols * scale, src.rows * scale));
    }else if(type == 1){
        pyrDown(src, dst, Size(src.cols / scale, src.rows / scale));
    }

    Mat src_gray, g1, g2, g3;
    cvtColor(src, src_gray, COLOR_BGR2GRAY);
    GaussianBlur(src_gray, g1, Size(5, 5), 0, 0);
    GaussianBlur(g1, g2, Size(5, 5), 0, 0);
    subtract(g1, g2, g3, Mat());
    normalize(g3, g3, 255, 0, NORM_MINMAX);
    imshow("DOG, difference of Gaussian", g3);

    namedWindow("zoom image", WINDOW_AUTOSIZE);
    imshow("zoom image", dst);
    waitKey(0);
}

void MyCV::threshold_operation() const{
    // define threshold parameters
    int threshold_value = 127;
    int threshold_max = 255;

    //define window
    namedWindow("threshold_demo", WINDOW_AUTOSIZE);
    createTrackbar("threshold value",
                   "threshold_demo",
                   &threshold_value,
                   threshold_max,
                   threshold_operation_callback,
                   cv_pointer);
    threshold_operation_callback(0, cv_pointer);
    waitKey(0);
}

void MyCV::threshold_operation_callback(int pos, void * userdata) {
    auto * myCanny = (MyCV *) userdata;
    Mat this_src = myCanny->src;

    // define Mat
    Mat gray_src;

    // change color version?
    cvtColor(this_src, gray_src, COLOR_RGB2GRAY);

    // threshold
    threshold(gray_src, myCanny->dst, pos, 255, THRESH_BINARY);
    imshow("threshold_demo", myCanny->dst);
}

void MyCV::self_defined_filter(int type) {

    Mat RobertX = (Mat_<int> (2, 2) << 1, 0, 0, -1);
    Mat RobertY = (Mat_<int> (2, 2) << 0, 1, -1, 0);
    Mat SobelX = (Mat_<int> (3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
    Mat SobelY = (Mat_<int> (3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, -1);
    Mat Laplace = (Mat_<int> (3, 3) << 0, -1, 0, -1, 4, -1, 0, -1, 0);

    switch (type){
        case 0:{
            filter2D(src, dst, -1, RobertX, Point(-1, -1));
            break;
        }
        case 1:{
            filter2D(src, dst, -1, RobertY, Point(-1, -1));
            break;
        }
        case 2:{
            filter2D(src, dst, -1, SobelX, Point(-1, -1));
            break;
        }
        case 3:{
            filter2D(src, dst, -1, SobelY, Point(-1, -1));
            break;
        }
        case 4:{
            filter2D(src, dst, -1, Laplace, Point(-1, -1));
            break;
        }
        default:{
            cout << "Error encountered! wrong type..." << endl;
            break;
        }
    }

    imshow("output", dst);
    waitKey(0);
}

void MyCV::add_edges() {
    int top = (int)(0.05 * src.rows);
    int bottom = (int)(0.05 * src.rows);
    int left = (int)(0.05 * src.cols);
    int right = (int)(0.05 * src.cols);
    RNG rng(12345);

    int border_type = BORDER_DEFAULT;

    int c = 0;
    while (true){
        c = waitKey(500);
        cout << "Be advised, press Esc to exit" << endl;
        if ((char)c == 27){
            break;
        }if ((char)c == 'r'){
            border_type = BORDER_REPLICATE;
        }if ((char)c == 'w'){
            border_type = BORDER_WRAP; // use another side
        }if ((char)c == 'c'){
            border_type = BORDER_CONSTANT;
        }else{
            border_type = BORDER_DEFAULT;
        }
        Scalar color = Scalar (rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        copyMakeBorder(src, dst, top, bottom, left, right, border_type, color);
        imshow("output", dst);
    }
}

void MyCV::sobel_operation() {
    Mat mat_blur, src_gray;
    GaussianBlur(src, mat_blur, Size(3, 3), 0, 0);
    cvtColor(mat_blur, src_gray, COLOR_BGR2GRAY);
    Mat grad_x, grad_y;
    Sobel(src_gray, grad_x, CV_16S, 1, 0, 3);
    Sobel(src_gray, grad_y, CV_16S, 0, 1, 3);
    convertScaleAbs(grad_x, grad_x);
    convertScaleAbs(grad_y, grad_y);

    Mat grad;
    addWeighted(grad_x, 0.5, grad_y, 0.5, 0, grad);
    namedWindow("sobel", WINDOW_AUTOSIZE);
    imshow("sobel", grad);
    waitKey(0);
}

void MyCV::laplace_operation() {
    // extract edges
    Mat img = src;
    Mat gray_src, final_image;
    GaussianBlur(img, final_image, Size(3, 3), 0, 0);
    cvtColor(final_image, gray_src, COLOR_BGR2GRAY);

    Laplacian(gray_src, final_image, CV_16S, 3);
    convertScaleAbs(final_image, final_image);

    threshold(final_image, final_image, 0, 255, THRESH_BINARY); // kill noise

    namedWindow("final", WINDOW_AUTOSIZE);
    imshow("final", final_image);
    waitKey(0);
}

void MyCV::canny_process() {

}

void MyCV::canny_threshold_callback() {

}

void MyCV::extract_lines_pro() {
    // Basic Hough Line Transform
    // use API HoughLineP, be reminded while applying HoughLine(if u wish)
    Mat img = src;
    Mat final_image, rgb_final, gray_img;

    Canny(img, gray_img, 50, 200, 3);
    cvtColor(gray_img, rgb_final, COLOR_GRAY2BGR);

    vector<Vec4i> Lines;  // define vector to hold points(x-axis and y-axis points)
    HoughLinesP(gray_img,
                Lines, 1, CV_PI/180,
                50,
                50,
                15); // maxLineGap is essential in line detecting!

    for (auto l : Lines){
        line(rgb_final,
             Point(l[0], l[1]),
             Point(l[2], l[3]),
             Scalar(0, 0, 255), 3,
             LINE_AA);
    }  // consider using auto loop instead

    imshow("canny edge detector", gray_img);
    imshow("lines", rgb_final);
    waitKey(0);
}

void MyCV::set_src(Mat &img) {
    src = img;
}

void MyCV::extract_circle() {
    // detecting circle by Hough
    cout << "Hough circle detecting is called..." << endl;
    string storePath = "D:/Personal Files/project files/PI-LAB/learn_slam/images/hough_circle.jpg";
    Mat img = src;
    Mat gray_img, dst_img, blur_img, final_img;

    // median filter, killing noise
    medianBlur(img, blur_img, 3);
    cvtColor(blur_img, gray_img, COLOR_BGR2GRAY);

    // Hough circle detecting
    vector<Vec3f> circlePoints;
    HoughCircles(gray_img, circlePoints,
                 HOUGH_GRADIENT, 1, 80,
                 100, 30, 30, 100); // detecting points of 8-gray
                 // https://docs.opencv.org/4.5.2/dd/d1a/group__imgproc__feature.html#ga47849c3be0d0406ad3ca45db65a25d2d
    img.copyTo(final_img);
    for (auto l : circlePoints){
        circle(final_img,
               Point(l[0], l[1]), l[2],
               Scalar(0, 0, 255), 2, LINE_AA);
        circle(final_img,
               Point(l[0], l[1]), 1,
               Scalar(0, 0, 255), 2, LINE_AA);
    } // suggest using int rather than float?
    // FIXME: 这参数也太难调了，如果在一个工程中，图片可能是动态的（用户输入的），该如何调参？？
    // minRadius/maxRadius 设置成default 0 ，运算会爆炸
    // show image
    imshow("blur_img", blur_img);
    imshow("gray_img", gray_img);
    imshow("final_image", final_img);
    waitKey(0);

    imwrite(storePath, final_img);
}

void MyCV::compose_remapping() {

}

void function1_10(){
    // 基本的形态学操作
    MyCV myCv;
    myCv.set_morphology_operator(2);
    myCv.morphology_operation();
    // 怎么感觉gradient梯度操作和canny边缘算子很像呢，注意原理
    // morphology_gradient = dilate - erode
}

void function1_11(){
    // 提取水平和垂直的线，利用形态学操作
    Mat img = imread("../../images/extract_hon_line.png");
    MyCV myCv(img);
    myCv.extract_lines();
}

void function1_12(){
    Mat img = imread("../../images/learn_slam_inked.jpg");
    MyCV myCv(img);
    myCv.erase_line(3);
}

void function1_13(){
    // 图像金字塔变换
    Mat img = imread("../../images/learn_slam_inked.jpg");
    MyCV myCv(img);
    myCv.image_pyramid(4, 1);
    // DOG 出来也有点canny算子的感觉
}

void function1_14(){
    MyCV myCv;
    myCv.threshold_operation();
    myCv.self_defined_filter(4);
    myCv.add_edges();
    myCv.sobel_operation();
    myCv.laplace_operation();
}

void function1_15(){
    MyCV myCv;
    Mat img = imread("../../images/sudoku.png");
    myCv.set_src(img);
    myCv.extract_lines_pro();
    // 找适合提取的图片挺难的
};

void function1_16(){
    // 圆检测，Hough circle
    Mat img = imread("../../images/coins.jpg");
    MyCV myCv(img);
    myCv.extract_circle();
}


