#include <iostream>
#include <string>

#include "mCode/test1.h"
#include "mCode/myEdge.h"

using namespace std;

void main_advanced(){
    // 命令行调用执行程序
    string head_name = "function";
    cout << "please choose work number:" ;
    string work_id;
    cin >> work_id;
    cout << "please choose code number:" ;
    string code_id;
    cin >> code_id;
    string work_name = head_name + work_id + "_" + code_id;
    // 例如依次输入，2,1，就代表运行function2_1，也就是work2-1的函数

    #define DEF_TEST(f) if ( work_name == #f ) return f();
    DEF_TEST(function1_0);
    DEF_TEST(function1_1);
    DEF_TEST(function1_2);
    DEF_TEST(function1_3);
    DEF_TEST(function1_4);
    DEF_TEST(function1_5);
    DEF_TEST(function1_6);
    DEF_TEST(function1_7);
    DEF_TEST(function1_8);
    DEF_TEST(function1_9);
    DEF_TEST(function1_10);
    DEF_TEST(function1_11);
    DEF_TEST(function1_12);
    DEF_TEST(function1_13);
    DEF_TEST(function1_14);
    DEF_TEST(function1_15);
    DEF_TEST(function1_16);
    DEF_TEST(function2_1);

}

int main() {
    main_advanced();
    std::cout << "file ending" << std::endl;
    return 0;
}
