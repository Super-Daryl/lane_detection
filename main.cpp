#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "LaneDetector.h"
#include"CardDetector.h"
using namespace std;
using namespace cv;

int main(){
    lane_detector();
    Card_Detector();
    return 1;
}

