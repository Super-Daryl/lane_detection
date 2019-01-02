#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "LaneDetector.h"
#include<math.h>
#define  PI 3.1415926
// IMAGE BLURRING
/**
 *@brief Apply gaussian filter to the input image to denoise it
 *@param inputImage is the frame of a video in which the
 *@param lane is going to be detected
 *@return Blurred and denoised image
 */
Mat LaneDetector::deNoise(Mat inputImage) {
    Mat output;

    GaussianBlur(inputImage, output, Size(3, 3), 0, 0);

    return output;
}

// EDGE DETECTION
/**
 *@brief Detect all the edges in the blurred frame by filtering the image
 *@param img_noise is the previously blurred frame
 *@return Binary image with only the edges represented in white
 */
Mat LaneDetector::edgeDetector(Mat img_noise) {
    Mat output;
    Mat kernel;
    Point anchor;

    // Convert image from RGB to gray
    cvtColor(img_noise, output, COLOR_RGB2GRAY);
      imshow ("gray",output);
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
//    filter2D(output, output, -1, kernel, anchor, 0, BORDER_DEFAULT);
    imshow("output", output);
    //d    waitKey(0);
    return output;
}

// MASK THE EDGE IMAGE
/**
 *@brief Mask the image so that only the edges that form part of the lane are detected
 *@param img_edges is the edges image from the previous function
 *@return Binary image with only the desired edges being represented
 */
//Mat& LaneDetector::CardDector (Mat &srcImg){
//    int width = srcImg.cols;//图像宽度
//    int height = srcImg.rows;//图像高度
//    double B = 0.0, G = 0.0, R = 0.0, H = 0.0, S = 0.0, V = 0.0;
//    Mat matRgb = Mat::zeros(srcImg.size(), CV_8UC1);
//    int x, y; //循环
//    for (y = 0; y < height; y++)
//    {
//        for (x = 0; x < width; x++)
//        {
//            // 获取BGR值
//            B = srcImg.at<Vec3b>(y, x)[0];
//            G = srcImg.at<Vec3b>(y, x)[1];
//            R = srcImg.at<Vec3b>(y, x)[2];
//            if (G-B>50&&G-R>30){
//                matRgb.at<uchar>(y, x) = 255;
//            }
////            RGB2HSV(R, G, B, H, S, V);
//            //红色范围
////            if ((H >= 330 && H <= 360 || H >= 0 && H <= 10) && S >= 21 && S <= 100 && V > 16 && V < 99) //H不能低于10，H不能大于344,S不能高于21，V不能变
////            {
////                matRgb.at<uchar>(y, x) = 255;
////            }
//        }
//    }
//    imshow("matRgb", matRgb);
////    waitKey(0);
//    return matRgb;
//}



Mat LaneDetector::mask(Mat img_edges) {
    Mat output;
    Mat mask = Mat::zeros(img_edges.size(), img_edges.type());
    Point pts[4] = {
        Point(210, 720),
        Point(550, 450),
        Point(717, 450),
        Point(1280, 720)
    };

    // Create a binary polygon mask
    fillConvexPoly(mask, pts, 4, Scalar(255, 0, 0));
      imshow("mask",mask);
    // Multiply the edges image and the mask to get the output
    bitwise_and(img_edges, mask, output);
    imshow("bitwise_and",output);
    return output;
}

// HOUGH LINES
/**
 *@brief Obtain all the line segments in the masked images which are going to be part of the lane boundaries
 *@param img_mask is the masked binary image from the previous function
 *@return Vector that contains all the detected lines in the image
 */
vector<Vec4i> LaneDetector::houghLines(Mat img_mask) {
    vector<Vec4i> line;

    // rho and theta are selected by trial and error
    HoughLinesP(img_mask, line, 1, CV_PI / 180, 20, 20, 30);

    return line;
}

// SORT RIGHT AND LEFT LINES
/**
 *@brief Sort all the detected Hough lines by slope.
 *@brief The lines are classified into right or left depending
 *@brief on the sign of their slope and their approximate location
 *@param lines is the vector that contains all the detected lines
 *@param img_edges is used for determining the image center
 *@return The output is a vector(2) that contains all the classified lines
 */
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
/**
 *@brief Regression takes all the classified line segments initial and final points and fits a new lines out of them using the method of least squares.
 *@brief This is done for both sides, left and right.
 *@param left_right_lines is the output of the lineSeparation function
 *@param inputImage is used to select where do the lines will end
 *@return output contains the initial and final points of both lane boundary lines
 */
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
    int fin_y = 470;

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

// TURN PREDICTION
/**
 *@brief Predict if the lane is turning left, right or if it is going straight

 *@return String that says if there is left or right turn or if the road is straight
 */
string LaneDetector::predictTurn() {
    string output;
    double vanish_x;
    double thr_vp = 10;

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
/**
 *@brief This function plots both sides of the lane, the turn prediction message and a transparent polygon that covers the area inside the lane boundaries
 *@param inputImage is the original captured frame
 *@param lane is the vector containing the information of both lines
 *@param turn is the output string containing the turn information
 *@return The function returns a 0
 */
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



//New
void LaneDetector::RGB2HSV(double red, double green, double blue, double& hue, double& saturation,double& intensity )
{
    double r,g,b;
    double h,s,i;
    double sum;
    double minRGB,maxRGB;
    double theta;
    r = red/255.0;
    g = green/255.0;
    b = blue/255.0;
    minRGB = ((r<g)?(r):(g));
    minRGB = (minRGB<b)?(minRGB):(b);
    maxRGB = ((r>g)?(r):(g));
    maxRGB = (maxRGB>b)?(maxRGB):(b);
    sum = r+g+b;
    i = sum/3.0;
    if( i<0.001 || maxRGB-minRGB<0.001 )
    {
        h=0.0;
        s=0.0;
    }
    else
    {
        s = 1.0-3.0*minRGB/sum;
        theta = sqrt((r-g)*(r-g)+(r-b)*(g-b));
        theta = acos((r-g+r-b)*0.5/theta);
        if(b<=g)
            h = theta;
        else
            h = 2*PI - theta;
        if(s<=0.01)
            h=0;
    }
    hue = (int)(h*180/PI);
    saturation = (int)(s*100);
    intensity = (int)(i*100);
}

void LaneDetector::fillHole(const Mat srcBw, Mat &dstBw)
{
    Size m_Size = srcBw.size();
    Mat Temp=Mat::zeros(m_Size.height+2,m_Size.width+2,srcBw.type());//延展图像
    srcBw.copyTo(Temp(Range(1, m_Size.height + 1), Range(1, m_Size.width + 1)));

    cv::floodFill(Temp, Point(0, 0), Scalar(255));//填充区域

    Mat cutImg;//裁剪延展的图像
    Temp(Range(1, m_Size.height + 1), Range(1, m_Size.width + 1)).copyTo(cutImg);

    dstBw = srcBw | (~cutImg);
}

int LaneDetector:: CardDector (Mat &src){
    //得到图像参数
    int width = src.cols; //图像宽度
    int height = src.rows; //图像高度
    //色彩分割
    double B=0.0,G=0.0,R=0.0,H=0.0,S=0.0,I=0.0,V=0.0;
    Mat Mat_rgb = Mat::zeros( src.size(), CV_8UC1 );
    int x,y; //循环
    RGB2HSV(49,116,109,H,S,I);
//    cout<<"h s i"<<H<<" "<<S<<" "<<I<<endl;
    for (y=0; y<height; y++)
    {
        for ( x=0; x<width; x++)
        {
            // 获取 BGR 值
            B = src.at<Vec3b>(y,x)[0];
            G = src.at<Vec3b>(y,x)[1];
            R = src.at<Vec3b>(y,x)[2];

            RGB2HSV(R,G,B,H,S,I);

            //红色：337-360
            //    if((H>=337 && H<=360||H>=0&&H<=10)&&S>=12&&S<=100&&V>20&&V<99)
            //    {
            //    Mat_rgb.at<uchar>(y,x) = 255; //分割出红色
            //    }
            if (160<H&&H<185&S>43&&S<255&&46<I&&I<255){
                Mat_rgb.at<uchar>(y,x) = 255; //分割出绿色
            }
        }
    }
//    imshow("Mat_rgb", Mat_rgb);

    Mat element = getStructuringElement(MORPH_RECT,
                                        Size(2*1 + 1, 2*1 + 1),
                                        Point(1,1));
    Mat element1 = getStructuringElement(MORPH_RECT,
                                         Size(2*3 + 1, 2*3 + 1),
                                         Point(3,3));
    //形态学处理图像，开操作和闭操作，消除噪声，连通有效区域
    Mat Mat_rgb_copy;
    //    erode(Mat_rgb,Mat_rgb,element);
    //     imshow("erode",Mat_rgb);
    dilate(Mat_rgb,Mat_rgb,element1);
//    imshow("dilate",Mat_rgb);
    fillHole(Mat_rgb,Mat_rgb);
//    imshow("fillhole",Mat_rgb);
    Mat_rgb.copyTo(Mat_rgb_copy);
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours( Mat_rgb, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
//    std::cout<<"size"<<contours.size()<<endl;
    for (size_t t=0;t<contours.size();t++){
        RotatedRect rect=minAreaRect(contours[t]);
        Point2f vertices[4];      //定义矩形的4个顶点
        rect.points(vertices);   //计算矩形的4个顶点
        rectangle(src, rect.boundingRect(), Scalar(255,0,0));
        //for (int i = 0; i < 4; i++){
        //  line(src, vertices[i], vertices[(i + 1) % 4], Scalar(255, 0, 0),2);
        //}
        //rectangle( src, rect[];, rect, color, 2, 8, 0 );
        // drawContours(src, rect, static_cast<int>(t), Scalar(255, 0, 0), 2, 8);
//        imshow("src",src);
    }
//    waitKey(0);
    return 1;
}
