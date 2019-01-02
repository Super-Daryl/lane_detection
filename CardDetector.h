#ifndef CARDDETECTOR_H
#define CARDDETECTOR_H
using namespace cv;

void RGB2HSV(double red, double green, double blue, double& hue, double& saturation, double& intensity);
void fillHole(const Mat srcBw, Mat &dstBw);
int Card_Detector();
#endif // CARDDETECTOR_H
