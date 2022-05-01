# LoopClosing
LoopClosing是专门负责做闭环的类，它的主要功能就是检测闭环，计算闭环帧的相对位姿病以此做闭环修正。

![image1](https://pic2.zhimg.com/80/v2-66d8860e4abbfa5824b59b5dafa6e445_720w.jpg)

## bool LoopClosing::DetectLoop()
言简意赅，返回一个bool类型变量，判断有无闭环；

