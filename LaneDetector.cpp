#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "LaneDetector.h"
#include<math.h>
#define  PI 3.1415926

Mat LaneDetector::deNoise(Mat inputImage) {
    Mat output;
    GaussianBlur(inputImage, output, Size(3, 3), 0, 0);
    return output;
}

// EDGE DETECTION
Mat LaneDetector::edgeDetector(Mat img_noise) {
    Mat output;
    Mat kernel;
    Point anchor;

    // Convert image from RGB to gray
    cvtColor(img_noise, output, COLOR_RGB2GRAY);
//  imshow ("gray",output);
    // Binarize gray image
    threshold(output, output, 140, 255, THRESH_BINARY);

//       threshold(output, output, 160, 255,  CV_THRESH_OTSU);
//    imshow(const String &winname, <#InputArray mat#>)

    // Create the kernel [-1 0 1]
    // This kernel is based on the one found in the
    // Lane Departure Warning System by Mathworks
    anchor = Point(-1, -1);
    kernel = Mat(1, 3, CV_32F);
    kernel.at<float>(0, 0) = -1;
    kernel.at<float>(0, 1) = 0;
    kernel.at<float>(0, 2) = 1;
//     imshow("threshold", output);
    // Filter the binary image to obtain the edges
    filter2D(output, output, -1, kernel, anchor, 0, BORDER_DEFAULT);
//    imshow("output", output);
    //    waitKey(0);
    return output;
}

Mat LaneDetector::mask(Mat img_edges) {
    Mat output;
    Mat mask = Mat::zeros(img_edges.size(), img_edges.type());
    Point pts[4] = {
        Point(210/2, 720/2),
        Point(550/2, 450/2),
        Point(717/2, 450/2),
        Point(1280/2, 720/2)
    };

    // Create a binary polygon mask
    fillConvexPoly(mask, pts, 4, Scalar(255, 0, 0));
//      imshow("mask",mask);
    // Multiply the edges image and the mask to get the output
    bitwise_and(img_edges, mask, output);
    imshow("bitwise_and",output);
    return output;
}

// HOUGH LINES
vector<Vec4i> LaneDetector::houghLines(Mat img_mask) {
    vector<Vec4i> line;

    // rho and theta are selected by trial and error
    HoughLinesP(img_mask, line, 1, CV_PI / 180, 20, 20, 30);

    return line;
}

// SORT RIGHT AND LEFT LINES
vector<vector<Vec4i> > LaneDetector::lineSeparation(vector<Vec4i> lines, Mat img_edges) {
    vector<vector<Vec4i> > output(2);
    size_t j = 0;
    Point ini;
    Point fini;
    double slope_thresh = 0.3;
    vector<double> slopes;
    vector<Vec4i> selected_lines;
    vector<Vec4i> right_lines, left_lines;

    // Calculate the slope of all the detected lines
    for (auto i : lines) {
        ini = Point(i[0], i[1]);
        fini = Point(i[2], i[3]);

        // Basic algebra: slope = (y1 - y0)/(x1 - x0)
        double slope = (static_cast<double>(fini.y) - static_cast<double>(ini.y)) / (static_cast<double>(fini.x) - static_cast<double>(ini.x) + 0.00001);

        // If the slope is too horizontal, discard the line
        // If not, save them  and their respective slope
        if (abs(slope) > slope_thresh) {
            slopes.push_back(slope);
            selected_lines.push_back(i);
        }
    }

    // Split the lines into right and left lines
    img_center = static_cast<double>((img_edges.cols / 2));
    while (j < selected_lines.size()) {
        ini = Point(selected_lines[j][0], selected_lines[j][1]);
        fini = Point(selected_lines[j][2], selected_lines[j][3]);

        // Condition to classify line as left side or right side
        if (slopes[j] > 0 && fini.x > img_center && ini.x > img_center) {
            right_lines.push_back(selected_lines[j]);
            right_flag = true;
        }
        else if (slopes[j] < 0 && fini.x < img_center && ini.x < img_center) {
            left_lines.push_back(selected_lines[j]);
            left_flag = true;
        }
        j++;
    }

    output[0] = right_lines;
    output[1] = left_lines;

    return output;
}

// REGRESSION FOR LEFT AND RIGHT LINES
vector<Point> LaneDetector::regression(vector<vector<Vec4i> > left_right_lines, Mat inputImage) {
    vector<Point> output(4);
    Point ini;
    Point fini;
    Point ini2;
    Point fini2;
    Vec4d right_line;
    Vec4d left_line;
    vector<Point> right_pts;
    vector<Point> left_pts;

    // If right lines are being detected, fit a line using all the init and final points of the lines
    if (right_flag == true) {
        for (auto i : left_right_lines[0]) {
            ini = Point(i[0], i[1]);
            fini = Point(i[2], i[3]);

            right_pts.push_back(ini);
            right_pts.push_back(fini);
        }

        if (right_pts.size() > 0) {
            // The right line is formed here
            fitLine(right_pts, right_line, CV_DIST_L2, 0, 0.01, 0.01);
            right_m = right_line[1] / right_line[0];
            right_b = Point(right_line[2], right_line[3]);
        }
    }

    // If left lines are being detected, fit a line using all the init and final points of the lines
    if (left_flag == true) {
        for (auto j : left_right_lines[1]) {
            ini2 = Point(j[0], j[1]);
            fini2 = Point(j[2], j[3]);

            left_pts.push_back(ini2);
            left_pts.push_back(fini2);
        }

        if (left_pts.size() > 0) {
            // The left line is formed here
            fitLine(left_pts, left_line, CV_DIST_L2, 0, 0.01, 0.01);
            left_m = left_line[1] / left_line[0];
            left_b = Point(left_line[2], left_line[3]);
        }
    }

    // One the slope and offset points have been obtained, apply the line equation to obtain the line points
    int ini_y = inputImage.rows;
    int fin_y = 470/2;

    double right_ini_x = ((ini_y - right_b.y) / right_m) + right_b.x;
    double right_fin_x = ((fin_y - right_b.y) / right_m) + right_b.x;

    double left_ini_x = ((ini_y - left_b.y) / left_m) + left_b.x;
    double left_fin_x = ((fin_y - left_b.y) / left_m) + left_b.x;

    output[0] = Point(right_ini_x, ini_y);
    output[1] = Point(right_fin_x, fin_y);
    output[2] = Point(left_ini_x, ini_y);
    output[3] = Point(left_fin_x, fin_y);

    return output;
}


string LaneDetector::predictTurn() {
    string output;
    double vanish_x;
    double thr_vp = 5;

    // The vanishing point is the point where both lane boundary lines intersect
    vanish_x = static_cast<double>(((right_m*right_b.x) - (left_m*left_b.x) - right_b.y + left_b.y) / (right_m - left_m));

    // The vanishing points location determines where is the road turning
    if (vanish_x < (img_center - thr_vp))
        output = "Left Turn";
    else if (vanish_x >(img_center + thr_vp))
        output = "Right Turn";
    else if (vanish_x >= (img_center - thr_vp) && vanish_x <= (img_center + thr_vp))
        output = "Straight";

    return output;
}

// PLOT RESULTS
int LaneDetector::plotLane(Mat inputImage, vector<Point> lane, string turn) {
    vector<Point> poly_points;
    Mat output;

    // Create the transparent polygon for a better visualization of the lane
    inputImage.copyTo(output);
    poly_points.push_back(lane[2]);
    poly_points.push_back(lane[0]);
    poly_points.push_back(lane[1]);
    poly_points.push_back(lane[3]);
    fillConvexPoly(output, poly_points, Scalar(0, 0, 255), CV_AA, 0);
    addWeighted(output, 0.3, inputImage, 1.0 - 0.3, 0, inputImage);

    // Plot both lines of the lane boundary
    line(inputImage, lane[0], lane[1], Scalar(0, 255, 255), 5, CV_AA);
    line(inputImage, lane[2], lane[3], Scalar(0, 255, 255), 5, CV_AA);

    // Plot the turn message
    putText(inputImage, turn, Point(50, 90), FONT_HERSHEY_COMPLEX_SMALL, 3, cvScalar(0, 255, 0), 1, CV_AA);

    // Show the final output image
    namedWindow("Lane", CV_WINDOW_AUTOSIZE);
    imshow("Lane", inputImage);
//    waitKey(0);
    return 0;
}

//Master function
int lane_detector() {

    // The input argument is the location of the video
    VideoCapture cap("/Users/machinelee/advanced_lane_detection/project_video.mp4");
//        VideoCapture cap("/Users/machinelee/Desktop/1.mp4");
    if (!cap.isOpened())
    {
        cout<<"can't laod"<<endl;
        return -1;
    }
    LaneDetector lanedetector;  // Create the class object
    Mat frame;
    Mat img_denoise;
    Mat img_edges;
    Mat img_mask;
    Mat img_lines;
    vector<Vec4i> lines;
    vector<vector<Vec4i> > left_right_lines;
    vector<Point> lane;
    string turn;
    int flag_plot = -1;
    int i = 0;
    // Main algorithm starts. Iterate through every frame of the video
    while (i < 540) {
        if (!cap.read(frame))
            break;
        resize(frame,frame,Size(frame.cols/2,frame.rows/2));
        Mat dest;

        //        frame=imread("/Users/machinelee/Desktop/3.png");
        // Denoise the image using a Gaussian filter
        //        lanedetector.CardDector(frame);
//        lanedetector.CardDector(frame)
        img_denoise = lanedetector.deNoise(frame);
            imshow("src",frame);


        // Detect edges in the image
        img_edges = lanedetector.edgeDetector(img_denoise);

        // Mask the image so that we only get the ROI
        img_mask = lanedetector.mask(img_edges);

        // Obtain Hough lines in the cropped image
        lines = lanedetector.houghLines(img_mask);

        if (!lines.empty())
        {
            // Separate lines into left and right lines
            left_right_lines = lanedetector.lineSeparation(lines, img_edges);

            // Apply regression to obtain only one line for each side of the lane
            lane = lanedetector.regression(left_right_lines, frame);

            // Predict the turn by determining the vanishing point of the the lines
            turn = lanedetector.predictTurn();

            // Plot lane detection
            flag_plot = lanedetector.plotLane(frame, lane, turn);

            i += 1;
            if(waitKey(25)==27)
                break;

        }
        else {
            flag_plot = -1;
        }
    }
    return flag_plot;
}



