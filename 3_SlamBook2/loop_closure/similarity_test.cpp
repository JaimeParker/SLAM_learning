//
// Created by hazyparker on 2022/4/13.
// this function is designed to detect similarity of 10 images from dataset TUM
// following naming style of google: https://google.github.io/styleguide/cppguide.html
// reference: SlamBook2, https://github.com/gaoxiang12/slambook2/blob/master/ch11/loop_closure.cpp
//

#include "similarity_test.h"
#include "DBoW3/DBoW3.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <vector>
#include <string>

using namespace std;
using namespace cv;

int main(int argc, char **argv){
    // read database(dictionary)
    cout << "Loading Vocabulary, this may take a while..." << endl;
    DBoW3::Vocabulary my_voc("../vocabulary.yml.gz");
    if (my_voc.empty()){
        cerr << "fatal: Vocabulary does not exist or empty" << endl;
        return 1;
    }
    cout << "Vocabulary loaded!" << endl;

    // read images
    cout << "Reading images..." << endl;
    vector<Mat> images;
    for (int i = 0; i < 10; i++){
        string path = "../data/" + to_string(i + 1) + ".png";
        images.push_back(imread(path));
    }
    cout << "Images loaded!" << endl;

    // detect ORB features of all images
    cout << "Detecting ORB features..." << endl;
    Ptr<Feature2D> detector = ORB::create();  // method to define detector of ORB
    vector<Mat> descriptors;                  // define descriptors, saving descriptors
    for (Mat &image:images){
        vector<KeyPoint> key_points;
        Mat descriptor;
        detector->detectAndCompute(image, Mat(), key_points, descriptor);
        descriptors.push_back(descriptor);    // push each descriptor into descriptors
    }

    // compare with database
    cout << "Comparing images with database" << endl;
    DBoW3::Database db(my_voc, false, 0);
    for (auto & descriptor : descriptors){
        db.add(descriptor);
    }
    cout << "Database info:" << db << endl;
    for (int i = 0; i < descriptors.size(); i++){
        DBoW3::QueryResults ret;  // ret = "n results" + <> etc
        db.query(descriptors[i], ret, 4);      // max result=4
        cout << "searching for image " << i << " returns " << ret << endl << endl;
    }
    cout << "Comparison done." << endl;

    return 0;
}


