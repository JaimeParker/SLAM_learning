# Canny Edge Detector类的Debug

Canny算子是OpenCV里面边缘检测的一个重要算子，可以在OpenCV的官方文件上查到其信息和示例代码[OpenCV: Canny Edge Detector](https://docs.opencv.org/4.5.2/da/d5c/tutorial_canny_detector.html)

看到之后，由于最近正在学C++的类，我就很想把它写成一个类`MyCanny`，形成如下效果

```c++
Mat img = imread(filename);
MyCanny myCanny(img);
myCanny.canny_process();
```

至此，输出图片，这样三行代码十分简洁且封装性好

于是我开始了尝试

## 1. 代码

**对类定义**

```c++
class MyCanny{
private:
    Mat src, src_gray;
    Mat dst, detected_edges; // input and output matrix
    int lowThreshold = 0;
    const int max_lowThreshold = 100;
    const int ratio = 3;
    const int kernel_size = 3;
    const char* window_name = "Edge Map"; // some parameters
    MyCanny * canny_pointer = this;

public:
    explicit MyCanny(const Mat &img); // 构造函数，用于对类的对象赋值，由于数据是private的，只能通过此种方式赋值
    MyCanny(); // 构造函数，用于初始化
    void canny_process(); // 用于进行主要的CannyEdge处理过程
    static void canny_threshold(int pos, void* userdata);
};
```

**方法定义**

```c++
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
```

这一套是可行的

## 2. Bug

### 2.1 `createTrackbar()`

[createTrackbar](https://docs.opencv.org/4.5.2/d7/dfc/group__highgui.html#gaf78d2155d30b728fc413803745b67a9b)

这个函数可以算是重中之重了，我当时就是没仔细看函数定义（看了，但没完全看），结果bug搞了两天

```c++

int cv::createTrackbar	(	const String & 	trackbarname,
	const String & 	winname,
	int * 	value,
	int 	count,
	TrackbarCallback 	onChange = 0,
	void * 	userdata = 0 
)	
```

Creates a trackbar and attaches it to the specified window.

The function createTrackbar creates a trackbar (a slider or range control) with the specified name and range, assigns a variable value to be a position synchronized with the trackbar and specifies the callback function onChange to be called on the trackbar position change. The created trackbar is displayed in the specified window winname.

作用是一个类似跟踪条的东西

Parameters

| trackbarname | Name of the created trackbar.                                |
| ------------ | ------------------------------------------------------------ |
| winname      | Name of the window that will be used as a parent of the created trackbar. |
| value        | Optional pointer to an integer variable whose value reflects the position of the slider. Upon creation, the slider position is defined by this variable. |
| count        | Maximal position of the slider. The minimal position is always 0. |
| onChange     | Pointer to the function to be called every time the slider changes position. This function should be prototyped as void Foo(int,void*); , where the first parameter is the trackbar position and the second parameter is the user data (see the next parameter). If the callback is the NULL pointer, no callbacks are called, but only value is updated. |
| userdata     | User data that is passed as is to the callback. It can be used to handle trackbar events without using global variables. |

解释一下，`onChange`是回调函数名，`userdata`是给用户的数据，这个参数很关键

* 这里要求回调函数必须是静态static的，不然会报错'Reference to non-static function member must be called'
* 但是将回调函数静态之后，该如何访问类的数据呢

还有，在这个函数之前必须有`namedWindow()`，我一开始没注意这种函数，总感觉这个``namedWindow`很没用，却不是；

### 2.2 `canny_threshold(int pos, void * userdata)`

* 第一个参数是trackbar的位置value，将会传到这个回调函数里面

* 第二个参数是用户参数，是从trackbar**最后一个参数指针**来的

而关于如何从静态成员中访问非静态成员数据，使用一个指向实例化类指针，可以参考

* [如何给一个回调函数传数据？-编程语言-CSDN问答](https://ask.csdn.net/questions/7504028?answer=53509292&spm=1001.2014.3001.5504)
* [类成员函数作为回调函数的方法及注意点_hanxiucaolss的博客-CSDN博客_类成员函数作为回调函数](https://blog.csdn.net/hanxiucaolss/article/details/89500738)
* [C++静态成员函数访问非静态成员的几种方法 - Ricky.K - 博客园 (cnblogs.com)](https://www.cnblogs.com/rickyk/p/4238380.html)
* 尤其是最后一个

```c++
auto * myCanny = (MyCanny *) userdata;
```

### 2.3 流程

* `createTrackbar`调用`canny_threshold`，并且通过最后一个参数指针，将数据传给`canny_threshold`
* `pos`是trackbar的`value`位置，把这个值赋给关键的`lowThreshold`
* 执行完一遍后，监测变化，如有变化，改变值，传输，回调，如此循环

## 3. 总结

* 看OpenCV官方文档本来是个挺好的学习方式的，但是进度慢了就着急了，想着直接用了，结果问题一大堆
* C++基础也不扎实，我查了好多关于this和回调函数，指针指向类的东西，类里面的数据访问
* 一路下来虽然低效，但是收获很多

*做项目要全面理解要实现什么功能，以及现有系统提供了什么功能及接口，然后才能实施，最后才是要对所用工具要熟悉，如你现在用的C++，缺一不可，否则做不出好的产品*



