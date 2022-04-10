# OpenCV Tutorials

name: ZH Liu

Student ID: 2018300485

Link: [OpenCV: OpenCV Tutorials](https://docs.opencv.org/4.5.2/d9/df8/tutorial_root.html)

> this is the tutorial of OpenCV4.5.2,  find the latest version on their website if you like





## 1. Introduction to OpenCV

**just introduce for windows**:

* Install
* Build applications with OpenCV inside Visual Studio
* Image Watch





## 2. The Core Functionality

namely, **core module**

### 2.0 Contents

contents below:

- [Mat - The Basic Image Container](https://docs.opencv.org/4.5.2/d6/d6d/tutorial_mat_the_basic_image_container.html)
- [How to scan images, lookup tables and time measurement with OpenCV](https://docs.opencv.org/4.5.2/db/da5/tutorial_how_to_scan_images.html)
- [Mask operations on matrices](https://docs.opencv.org/4.5.2/d7/d37/tutorial_mat_mask_operations.html)
- [Operations with images](https://docs.opencv.org/4.5.2/d5/d98/tutorial_mat_operations.html)
- [Adding (blending) two images using OpenCV](https://docs.opencv.org/4.5.2/d5/dc4/tutorial_adding_images.html)
- [Changing the contrast and brightness of an image!](https://docs.opencv.org/4.5.2/d3/dc1/tutorial_basic_linear_transform.html)
- [Discrete Fourier Transform](https://docs.opencv.org/4.5.2/d8/d01/tutorial_discrete_fourier_transform.html)
- [File Input and Output using XML and YAML files](https://docs.opencv.org/4.5.2/dd/d74/tutorial_file_input_output_with_xml_yml.html)
- [How to use the OpenCV parallel_for_ to parallelize your code](https://docs.opencv.org/4.5.2/d7/dff/tutorial_how_to_use_OpenCV_parallel_for_.html)

### 2.1 Mat-the basic image container

#### 2.1.1 Goal

a matrix containing all the intensity values of the pixel points

what digital devices get: numerical **values** for each of the **points** of the image

> similar to key-value in SQL, it's more like position-value

OpenCV is to process and manipulate the matrix information

#### 2.1.2 Mat

Mat: a class with two data parts

* the matrix header
* a pointer to the matrix containing the pixel values

```c++
Mat A;
A = imread(argv[1], IMREAD_COLOR);
// methods to copy a Mat(or part of it)
Mat B(A);
C = A;
Mat F = A.clone();
Mat G;
A.copyTo(G);
Mat D(A, Rect(x, y, width, height)); // x coordinate of top-left corner
Mat E = A(Range::all(), Range(1, 10)); // using row and column boundaries
```

#### 2.1.3 Storing methods

select color space and data type

* color space: RGB/HLS/CRCB/CIE
* data type: char/unsigned char/float/double

Increasing the size of component **also increases** the size of the whole picture in the memory

#### 2.1.4 Creating a Mat object explicitly

* Mat Constructor

```c++
Mat M(rows, cols, data_type, scalar);
/**
for data_type, they are basiclly like CV_xxx
CV_[the number of bits per item][Singed or Unsigned][Type of prefix] C [the channel number]
for scalar(B, G, R) means 3 channels and BGR(color space is RGB)
*/
```

* Arrays initializing

```c++
int sz[3] = {2, 2, 2};
Mat L(3, sz, CV_8UC(1), Scalar::all(0));
/**
dimension = 3
size of each dimension = 2, 2, 2
*/
```

* Mat create

```c++
Mat F;
F.create(4, 4, CV_8UC(2));
// dafult 205
```

* MATLAB style

```c++
Mat E = Mat::eye(4, 4, CV_64F);
Mat O = Mat::ones(2, 2, CV_32F);
Mat Z = Mat::zeros(3, 3, CV_8UC1);
```

#### 2.1.5 Output formatting

refer when using



### 2.2 How to scan images, lookup tables and time measurement with OpenCV

#### 2.2.1 How is the image matrix stored in memory?

tow channels BGR color system:

<img src="https://docs.opencv.org/master/tutorial_how_matrix_stored_2.png">

#### 2.2.2 The core function

the main idea is to compose a table, performing modification usage on an image, in OpenCV we use LUT

```c++
LUT(src, lookUpTable, dst);
```

`lookUpTable`, usually built as `Mat` type

```c++
uchar table[256]; // also need edit table
Mat lookUpTable(1, 256, CV_8U);
uchar *p = lookUpTable.ptr();
for (int i = 0; i < 256; ++i){
    p[i] = table[i];
}
```

`table` used as LUT table, you can see this word frequently if you've ever used PR, it can also modify video if you like; and table needs to be edited by users

more LUT information,  [click here](https://docs.opencv.org/master/d2/de8/group__core__array.html#gab55b8d062b7f5587720ede032d34156f), and some document below

| P    | Info                                                         |
| ---- | ------------------------------------------------------------ |
| src  | input array of 8-bit elements.                               |
| lut  | look-up table of 256 elements; in case of multi-channel input array, the table should either have a single channel (in this case the same table is used for all channels) or the same number of channels as in the input array. |
| dst  | output array of the same size and number of channels as src, and the same depth as lut. |



### 2.3 Mask operations on matrices

$$
I(i,j)=5I(i,j)-[I(i,j-1)+I(i,j+1)+I(i-1,j)+I(i+1,j)]
$$

#### 2.3.1 The basic method

```c++
void Sharpen(const Mat& myImage,Mat& Result)
{
    CV_Assert(myImage.depth() == CV_8U);  // accept only uchar images
    const int nChannels = myImage.channels();
    Result.create(myImage.size(),myImage.type());
    for(int j = 1 ; j < myImage.rows-1; ++j)
    {
        const uchar* previous = myImage.ptr<uchar>(j - 1);
        const uchar* current  = myImage.ptr<uchar>(j    );
        const uchar* next     = myImage.ptr<uchar>(j + 1);
        uchar* output = Result.ptr<uchar>(j);
        for(int i= nChannels;i < nChannels*(myImage.cols-1); ++i)
        {
            *output++ = saturate_cast<uchar>(5*current[i]
                         -current[i-nChannels] - current[i+nChannels] - previous[i] - next[i]);
        }
    }
    Result.row(0).setTo(Scalar(0));
    Result.row(Result.rows-1).setTo(Scalar(0));
    Result.col(0).setTo(Scalar(0));
    Result.col(Result.cols-1).setTo(Scalar(0));
}
```

#### 2.3.2 The filter2D function

kernel, namely mask, a matrix with values symbolizing the filter situation

```c++
    Mat kernel = (Mat_<char>(3,3) <<  0, -1,  0,
                                   -1,  5, -1,
                                    0, -1,  0);
    filter2D( src, dst1, src.depth(), kernel );
```



### 2.4 Operations with images

#### 2.4.1 Input/Output

load an image from a file and grayscale

```c++
Mat img = imread(filename);
Mat img = imread(filename, IMREAD_GRAYSCALE);
```

for a jpg file, a **3 channel image** is created by default; *please pay attention to channels*

```c++
imwrite(filename, image);
```

>`imencode` and `imdecode`
>
>to read or write an image from/to memory rather than a file

#### 2.4.2 Basic operations with images

**accessing pixel intensity values**

what need to know about an image:

* type of an image
* the number of channels

for one channel

```c++
            Scalar intensity = img.at<uchar>(y, x);
            Scalar intensity = img.at<uchar>(Point(x, y));
```

for BGR, 3 channels

```c++
            Vec3b intensity = img.at<Vec3b>(y, x);
            uchar blue = intensity.val[0];
            uchar green = intensity.val[1];
            uchar red = intensity.val[2];
```

> Note:
>
> uchar type is hex, return hexadecimal values
>
> so we need to transfer it to `int` type or `cout << hex << red;` like this

**memory management and reference counting**

`Point` method:

* `Point`+`number`+`type`
* `Point3f` means 3 dimension with float type (32 bit)

```c++
std::vector<Point3f> points;
// .. fill the array
Mat pointsMat = Mat(points).reshape(1);
// get a 32FC1 matrix with 3 colunms, not 32FC3 with 1 colunm
```

`cv::Sobel`

**primitive operations**

`Scalar`

```c++
typedef struct Scalar{
    double val[4];
}
```

array with 4 elements mostly 

`Rect`

create an area

```c++
            Rect r(10, 10, 100, 100);
            Mat smallImg = img(r);
```

`cvtColor`

Converts an image from one color space to another

```c++
        Mat img = imread("image.jpg"); // loading a 8UC3 image
        Mat grey;
        cvtColor(img, grey, COLOR_BGR2GRAY);
```

`convertTo`

```c++
        src.convertTo(dst, CV_32F);
```

**visualizing images**

`namedWindow` : compose a new window

`imshow` : show the image through the window name

`waitKey` : make sure the image shown can hold on and preserved by human being



### 2.5 Adding (blending) two images using OpenCV

#### 2.5.1 Theory

*linear blend operation*
$$
g(x)=(1-\alpha)f_0(x)+\alpha f_1(x)
$$
By varying alpha from 0 to 1 can be used to perform a temporal cross-dissolve between 2 images or videos

#### 2.5.2 Source code

#### 2.5.3 Explanation

`addWeighted()`

```c++
addWeighted(InputArray src1, double alpha, InputArray src2,
                              double beta, double gamma, OutputArray dst, int dtype = -1);
```

$\beta = 1-\alpha$, the equation is basically the same with 2.5.1; usually $\gamma=0$ 
$$
g(x)=\alpha f_0(x)+\beta f_1(x)+\gamma
$$
be advised that src1 and src2 should own the **same size and type**



### 2.6 Changing the contrast and brightness of an image!

#### 2.6.1 Theory

Two commonly used point processes are multiplication and addition with a constant; If think $f(x)$as the source image pixels and $g(x)$ the output image pixels, we can write the expression as below:
$$
g(x)=\alpha f(x) +\beta\\
g(i,j)=\alpha f(i,j)+\beta
$$

#### 2.6.2 Code

```c++
    for( int y = 0; y < image.rows; y++ ) {
        for( int x = 0; x < image.cols; x++ ) {
            for( int c = 0; c < image.channels(); c++ ) {
                new_image.at<Vec3b>(y,x)[c] =
                  saturate_cast<uchar>( alpha*image.at<Vec3b>(y,x)[c] + beta );
            }
        }
    }
```

#### 2.6.3 Explanation

`saturate_cast<uchar>`

Pixel values outside of the [0 ; 255] range will be saturated (i.e. a pixel value higher (/ lesser) than 255 (/ 0) will be clamped to 255 (/ 0)).

`convertTo()`

```c++
image.convertTo(new_image, -1, alpha, beta);
```

Both methods give the same result but `convertTo` is more optimized and works a lot faster

#### 2.6.4 Gamma correction

[Gamma correction - Wikipedia](https://en.wikipedia.org/wiki/Gamma_correction)

Gamma correction can be used to correct the brightness of an image by using a non linear transformation between the input values and the mapped output values:
$$
O=(\frac{I}{255})^\gamma \times 255
$$
![image](https://docs.opencv.org/master/Basic_Linear_Transform_Tutorial_gamma.png)

So the expression will be:
$$
g(i, j)=255\alpha (\frac{f(i,j)}{255})^\gamma+\beta
$$




### 2.7 Discrete Fourier Transform

#### 2.7.1 Source code

```c++
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
using namespace cv;
using namespace std;
static void help(char ** argv)
{
    cout << endl
        <<  "This program demonstrated the use of the discrete Fourier transform (DFT). " << endl
        <<  "The dft of an image is taken and it's power spectrum is displayed."  << endl << endl
        <<  "Usage:"                                                                      << endl
        << argv[0] << " [image_name -- default lena.jpg]" << endl << endl;
}
int main(int argc, char ** argv)
{
    help(argv);
    const char* filename = argc >=2 ? argv[1] : "lena.jpg";
    Mat I = imread( samples::findFile( filename ), IMREAD_GRAYSCALE);
    if( I.empty()){
        cout << "Error opening image" << endl;
        return EXIT_FAILURE;
    }
    Mat padded;                            //expand input image to optimal size
    int m = getOptimalDFTSize( I.rows );
    int n = getOptimalDFTSize( I.cols ); // on the border add zero values
    copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));
    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complexI;
    merge(planes, 2, complexI);         // Add to the expanded another plane with zeros
    dft(complexI, complexI);            // this way the result may fit in the source matrix
    // compute the magnitude and switch to logarithmic scale
    // => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
    split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
    Mat magI = planes[0];
    magI += Scalar::all(1);                    // switch to logarithmic scale
    log(magI, magI);
    // crop the spectrum, if it has an odd number of rows or columns
    magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));
    // rearrange the quadrants of Fourier image  so that the origin is at the image center
    int cx = magI.cols/2;
    int cy = magI.rows/2;
    Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
    Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
    Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right
    Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);
    normalize(magI, magI, 0, 1, NORM_MINMAX); // Transform the matrix with float values into a
                                            // viewable image form (float between values 0 and 1).
    imshow("Input Image"       , I   );    // Show the result
    imshow("spectrum magnitude", magI);
    waitKey();
    return EXIT_SUCCESS;
}
```

#### 2.7.2 Explanation

*Fourier transform:*
$$
F(k,l)=\sum_{i=0}^{N-1} \sum_{j=0}^{N-1}f(i,j)e^{-2i\pi (\frac{ki}{N}+\frac{lj}{N})}\\
e^{ix}=\cos x+i \sin x
$$

* the DFT, `dft()`, will transfer an image from its **spatial domain to its frequency domain**

* magnitude image is interesting because it contains the information we need about images **geometric structure**

here are the steps:

**Expand the image to an optimal size**

```c++
    Mat padded;                            //expand input image to optimal size
	// Mat I before
    int m = getOptimalDFTSize( I.rows );
    int n = getOptimalDFTSize( I.cols ); // on the border add zero values
    copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));
// padder is the dst
```

`getOptimalDFTSize()`

* returns the optimal size

* to achieve maximal performance, we need to pad border values to the image to get a size with such traits

[`copyMakeBorder()`](https://docs.opencv.org/master/d2/de8/group__core__array.html#ga2ac1049c2c3dd25c2b41bffe17658a36)

* expand borders of an image
* the appended pixels are initialized with zero

**Make place for both the complex and the real values**

```c++
    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complexI;
    merge(planes, 2, complexI);         // Add to the expanded another plane with zeros
```

expand with another channel to hold complex values

[`merge`](https://docs.opencv.org/master/d2/de8/group__core__array.html#ga7d7b4d6c6ee504b30a20b1680029c7b4)

* Creates one multi-channel array out of several single-channel ones

**Make the discrete transform**

```c++
    dft(complexI, complexI);            // this way the result may fit in the source matrix
```

**Transfer the real and complex values to magnitude**
$$
M=\sqrt{Re(DFT(I))^2+Im(DFT(I))^2}
$$

```c++
    split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
    Mat magI = planes[0];
```

[`spilt`](https://docs.opencv.org/master/d2/de8/group__core__array.html#ga0547c7fed86152d7e9d0096029c8518a)

* contrast to `merge`, the function divides a multi-channel array into several single-channel arrays

[`magnitude`](https://docs.opencv.org/master/d2/de8/group__core__array.html#ga6d3b097586bca4409873d64a90fe64c3)

* Calculates the magnitude of 2D vectors
* the last parameter is output array

**Switch to logarithmic scale**

It turns out that the dynamic range of the Fourier coefficients is too large to be displayed on the screen, need to observe like this
$$
M_1=\log(1+M)
$$
in case M=0

```c++
    magI += Scalar::all(1);                    // switch to logarithmic scale
    log(magI, magI);
```

[`log`](https://docs.opencv.org/master/d0/de1/group__core.html#ga4eba02a849f926ee1764acde47108753)

**Crop and rearrange**

to throw away the newly introduced values(expanded values) and rearrange the quadrants of the result

```c++
    // crop the spectrum, if it has an odd number of rows or columns
    magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));
    // rearrange the quadrants of Fourier image  so that the origin is at the image center
    int cx = magI.cols/2;
    int cy = magI.rows/2;
    Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
    Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
    Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right
    Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);
```

**Normalize**

normalize display values to range of zero to one

```c++
    normalize(magI, magI, 0, 1, NORM_MINMAX); // Transform the matrix with float values into a
                                            // viewable image form (float between values 0 and 1).
```

#### 2.7.3 Result

the physics meaning of DFT-[reference](https://www.cnblogs.com/tenderwx/p/5245859.html)



### 2.8 [File Input and Output using XML and YAML files](https://docs.opencv.org/4.5.2/dd/d74/tutorial_file_input_output_with_xml_yml.html)

### 2.9 [How to use the OpenCV parallel_for_ to parallelize your code](https://docs.opencv.org/4.5.2/d7/dff/tutorial_how_to_use_OpenCV_parallel_for_.html)

multithreading

CUDA?





## 3. Image Processing

### 3.0 Contents

#### Basic

- [Basic Drawing](https://docs.opencv.org/master/d3/d96/tutorial_basic_geometric_drawing.html)
- [Random generator and text with OpenCV](https://docs.opencv.org/master/df/d61/tutorial_random_generator_and_text.html)
- [Smoothing Images](https://docs.opencv.org/master/dc/dd3/tutorial_gausian_median_blur_bilateral_filter.html)
- [Eroding and Dilating](https://docs.opencv.org/master/db/df6/tutorial_erosion_dilatation.html)
- [More Morphology Transformations](https://docs.opencv.org/master/d3/dbe/tutorial_opening_closing_hats.html)
- [Hit-or-Miss](https://docs.opencv.org/master/db/d06/tutorial_hitOrMiss.html)
- [Extract horizontal and vertical lines by using morphological operations](https://docs.opencv.org/master/dd/dd7/tutorial_morph_lines_detection.html)
- [Image Pyramids](https://docs.opencv.org/master/d4/d1f/tutorial_pyramids.html)
- [Basic Thresholding Operations](https://docs.opencv.org/master/db/d8e/tutorial_threshold.html)
- [Thresholding Operations using inRange](https://docs.opencv.org/master/da/d97/tutorial_threshold_inRange.html)

#### Transformations

- [Making your own linear filters!](https://docs.opencv.org/master/d4/dbd/tutorial_filter_2d.html)
- [Adding borders to your images](https://docs.opencv.org/master/dc/da3/tutorial_copyMakeBorder.html)
- [Sobel Derivatives](https://docs.opencv.org/master/d2/d2c/tutorial_sobel_derivatives.html)
- [Laplace Operator](https://docs.opencv.org/master/d5/db5/tutorial_laplace_operator.html)
- [Canny Edge Detector](https://docs.opencv.org/master/da/d5c/tutorial_canny_detector.html)
- [Hough Line Transform](https://docs.opencv.org/master/d9/db0/tutorial_hough_lines.html)
- [Hough Circle Transform](https://docs.opencv.org/master/d4/d70/tutorial_hough_circle.html)
- [Remapping](https://docs.opencv.org/master/d1/da0/tutorial_remap.html)
- [Affine Transformations](https://docs.opencv.org/master/d4/d61/tutorial_warp_affine.html)

#### Histograms

- [Histogram Equalization](https://docs.opencv.org/master/d4/d1b/tutorial_histogram_equalization.html)
- [Histogram Calculation](https://docs.opencv.org/master/d8/dbc/tutorial_histogram_calculation.html)
- [Histogram Comparison](https://docs.opencv.org/master/d8/dc8/tutorial_histogram_comparison.html)
- [Back Projection](https://docs.opencv.org/master/da/d7f/tutorial_back_projection.html)
- [Template Matching](https://docs.opencv.org/master/de/da9/tutorial_template_matching.html)

#### Contours

- [Finding contours in your image](https://docs.opencv.org/master/df/d0d/tutorial_find_contours.html)
- [Convex Hull](https://docs.opencv.org/master/d7/d1d/tutorial_hull.html)
- [Creating Bounding boxes and circles for contours](https://docs.opencv.org/master/da/d0c/tutorial_bounding_rects_circles.html)
- [Creating Bounding rotated boxes and ellipses for contours](https://docs.opencv.org/master/de/d62/tutorial_bounding_rotated_ellipses.html)
- [Image Moments](https://docs.opencv.org/master/d0/d49/tutorial_moments.html)
- [Point Polygon Test](https://docs.opencv.org/master/dc/d48/tutorial_point_polygon_test.html)

#### Others

- [Image Segmentation with Distance Transform and Watershed Algorithm](https://docs.opencv.org/master/d2/dbd/tutorial_distance_transform.html)
- [Out-of-focus Deblur Filter](https://docs.opencv.org/master/de/d3c/tutorial_out_of_focus_deblur_filter.html)
- [Motion Deblur Filter](https://docs.opencv.org/master/d1/dfd/tutorial_motion_deblur_filter.html)
- [Anisotropic image segmentation by a gradient structure tensor](https://docs.opencv.org/master/d4/d70/tutorial_anisotropic_image_segmentation_by_a_gst.html)
- [Periodic Noise Removing Filter](https://docs.opencv.org/master/d2/d0b/tutorial_periodic_noise_removing_filter.html)

### 3.1 Basic drawing

#### 3.1.1 OpenCV Theory

`Point`

It represents a 2D point, specified by its image coordinates x and y. We can define it as:

```c++
Point pt;
pt.x = 10;
pt.y = 8;
```

or

```c++
Point pt =  Point(10, 8);
```

[`Scalar`](https://docs.opencv.org/master/dc/d84/group__core__basic.html#ga599fe92e910c027be274233eccad7beb)

- Represents a 4-element vector. The type Scalar is widely used in OpenCV for passing pixel values.
- In this tutorial, we will use it extensively to represent BGR color values (3 parameters). It is not necessary to define the last argument if it is not going to be used.
- Let's see an example, if we are asked for a color argument and we give:

```c++
Scalar(B, G, R)
```

#### 3.1.2 Code

 [Github content](https://raw.githubusercontent.com/opencv/opencv/master/samples/cpp/tutorial_code/ImgProc/basic_drawing/Drawing_1.cpp)

#### 3.1.3 Explanation



### 3.2 Random generator and text with OpenCV

### 3.3 Smoothing images

### 3.18 Remapping

#### 3.18.1 Theory

$$
g(x,y)=f(h(x,y))
$$

### 3.23 Back Projection

#### 3.23.1 Theory

Back Projection is a way of recording how well the pixels of a given image fit the distribution of pixels in a histogram model;

To put it simple, you calculate the histogram model of a feature and then use it to find this feature in an image;

[reference_CH](https://blog.csdn.net/shuiyixin/article/details/80331839)

use `HSV`, normally use `HS`(Hue and Saturation);

#### 3.23.2 Code

[Code Demo on Github](https://github.com/opencv/opencv/blob/master/samples/cpp/tutorial_code/Histograms_Matching/calcBackProject_Demo1.cpp)

#### 3.23.3 Explanaiton

- Loads an image
- Convert the original to HSV format and separate only *Hue* channel to be used for the Histogram (using the OpenCV function [cv::mixChannels](https://docs.opencv.org/4.5.3/d2/de8/group__core__array.html#ga51d768c270a1cdd3497255017c4504be) )
- Let the user to enter the number of bins to be used in the calculation of the histogram.
- Calculate the histogram (and update it if the bins change) and the backprojection of the same image.
- Display the backprojection and the histogram in windows.

**Transform it to HSV format:**

```cpp
    Mat hsv;
    cvtColor( src, hsv, COLOR_BGR2HSV );
```

**Separate Hue channel:**

```cpp
    hue.create(hsv.size(), hsv.depth());
    int ch[] = { 0, 0 };
    mixChannels( &hsv, 1, &hue, 1, ch, 1 );
```

`mixChannels`: as it can be referred [here](https://docs.opencv.org/4.5.3/d2/de8/group__core__array.html#ga51d768c270a1cdd3497255017c4504be)

| Param  | description                                                  |
| ------ | ------------------------------------------------------------ |
| src    | input array or vector of matrices; all of the matrices must have the same size and the same depth. |
| nsrcs  | number of matrices in `src`.                                 |
| dst    | output array or vector of matrices; all the matrices **must be allocated**; their size and depth must be the same as in `src[0]`. |
| ndsts  | number of matrices in `dst`.                                 |
| fromTo | array of index pairs specifying which channels are copied and where; fromTo[2k] is a 0-based index of the input channel in src, fromTo[2k+1] is an index of the output channel in dst; the continuous channel numbering is used: the first input image channels are indexed from 0 to src[0].channels()-1, the second input image channels are indexed from src[0].channels() to src[0].channels() + src[1].channels()-1, and so on, the same scheme is used for the output image channels; as a special case, when fromTo[k*2] is negative, the corresponding output channel is filled with zero . |
| npairs | number of index pairs in `fromTo`.                           |

talking about parameters here:

- **&hsv:** The source array from which the channels will be copied
- **1:** The number of source arrays
- **&hue:** The destination array of the copied channels
- **1:** The number of destination arrays
- **ch[] = {0,0}:** The array of index pairs indicating how the channels are copied. In this case, the Hue(0) channel of &hsv is being copied to the 0 channel of &hue (1-channel)
- **1:** Number of index pairs

also there is an example:

```cpp
Mat bgra( 100, 100, CV_8UC4, Scalar(255,0,0,255) );
Mat bgr( bgra.rows, bgra.cols, CV_8UC3 );
Mat alpha( bgra.rows, bgra.cols, CV_8UC1 );
// forming an array of matrices is a quite efficient operation,
// because the matrix data is not copied, only the headers
Mat out[] = { bgr, alpha };
// bgra[0] -> bgr[2], bgra[1] -> bgr[1],
// bgra[2] -> bgr[0], bgra[3] -> alpha[0]
int from_to[] = { 0,2, 1,1, 2,0, 3,3 };
mixChannels( &bgra, 1, out, 2, from_to, 4 );
```

the code above should probably means copying `hsv[0]`to`hue[0]`;

**create a trackbar:**

```cpp
    const char* window_image = "Source image";
    namedWindow( window_image );
    createTrackbar("* Hue  bins: ", window_image, &bins, 180, Hist_and_Backproj );
    Hist_and_Backproj(0, 0);
```

**Hist_and_Backproj()**:

initialize parameters for [`calcHist`](https://docs.opencv.org/4.5.3/d6/dc7/group__imgproc__hist.html#ga4b2b5fd75503ff9e6844cc4dcdaed35d)

```cpp
    int histSize = MAX( bins, 2 );
    float hue_range[] = { 0, 180 };
    const float* ranges[] = { hue_range };
```

calculate Histogram and normalize it

```cpp
    Mat hist;
    calcHist( &hue, 1, 0, Mat(), hist, 1, &histSize, ranges, true, false );
    normalize( hist, hist, 0, 255, NORM_MINMAX, -1, Mat() );
```

* `histSize` is the value defined by user, tracking that bar, which means  *Array of histogram sizes in each dimension*

Get the Backprojection of the same image by calling the function [cv::calcBackProject](https://docs.opencv.org/4.5.3/d6/dc7/group__imgproc__hist.html#ga3a0af640716b456c3d14af8aee12e3ca)

```cpp
    Mat backproj;
    calcBackProject( &hue, 1, 0, hist, backproj, ranges, 1, true );
```

### 3.24 Template Matching



## 4. Application utils

