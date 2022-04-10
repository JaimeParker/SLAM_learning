#include <iostream>
#include <string>

using namespace std;

// include file headers
#include "gaussNewton.h"
#include "LMMethod.h"
#include "LMMethod2.h"
#include "ceresCurveFitting.h"
#include "g2oCurveFitting.h"

void main_advanced(){
    // calling function by name
    string work_name;
    cout << "type in here:" << endl;
    cin >> work_name;
#define DEF_TEST(f) if ( work_name == #f ) return f();
    DEF_TEST(GaussNewtonSolve);
    DEF_TEST(TrustRegionMethod);
    DEF_TEST(LMSolve2);
    DEF_TEST(CeresSolve);
    DEF_TEST(G2OSolve);

}

int main() {
    main_advanced();
    std::cout << "File ended" << std::endl;
    return 0;
}
