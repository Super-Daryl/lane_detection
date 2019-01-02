#ifndef LaneDetector_h
#define LaneDetector_h
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

class LaneDetector
{
private:
    double img_size;
    double img_center;
    bool left_flag = false;  // Tells us if there's left boundary of lane detected
    bool right_flag = false;  // Tells us if there's right boundary of lane detected
    Point right_b;  // Members of both line equations of the lane boundaries:
    double right_m;  // y = m*x + b
    Point left_b;  //
    double left_m;  //

public:
    Mat deNoise(Mat inputImage);  // Apply Gaussian blurring to the input Image
    Mat edgeDetector(Mat img_noise);  // Filter the image to obtain only edges
    Mat mask(Mat img_edges);  // Mask the edges image to only care about ROI
    vector<Vec4i> houghLines(Mat img_mask);  // Detect Hough lines in masked edges image
    vector<vector<Vec4i> > lineSeparation(vector<Vec4i> lines, Mat img_edges);  // Sprt detected lines by their slope into right and left lines
    vector<Point> regression(vector<vector<Vec4i> > left_right_lines, Mat inputImage);  // Get only one line for each side of the lane
    string predictTurn();  // Determine if the lane is turning or not by calculating the position of the vanishing point
    int plotLane(Mat inputImage, vector<Point> lane, string turn);  // Plot the resultant lane and turn prediction in the frame.
};
int lane_detector();
#endif /* LaneDetector_h */
