#include<opencv2/opencv.hpp>
#include<iostream>
#include"CardDetector.h"
#define PI 3.1415926

using namespace std;
using namespace cv;

void RGB2HSV(double red, double green, double blue, double& hue, double& saturation, double& intensity)
{

    double r, g, b;
    double h, s, i;

    double sum;
    double minRGB, maxRGB;
    double theta;

    r = red / 255.0;
    g = green / 255.0;
    b = blue / 255.0;

    minRGB = ((r<g) ? (r) : (g));
    minRGB = (minRGB<b) ? (minRGB) : (b);

    maxRGB = ((r>g) ? (r) : (g));
    maxRGB = (maxRGB>b) ? (maxRGB) : (b);

    sum = r + g + b;
    i = sum / 3.0;

    if (i<0.001 || maxRGB - minRGB<0.001)
    {
        h = 0.0;
        s = 0.0;
    }
    else
    {
        s = 1.0 - 3.0*minRGB / sum;
        theta = sqrt((r - g)*(r - g) + (r - b)*(g - b));
        theta = acos((r - g + r - b)*0.5 / theta);
        if (b <= g)
            h = theta;
        else
            h = 2 * PI - theta;
        if (s <= 0.01)
            h = 0;
    }

    hue = (int)(h * 180 / PI);
    saturation = (int)(s * 100);
    intensity = (int)(i * 100);
}

void fillHole(const Mat srcBw, Mat &dstBw)
{
    Size m_Size = srcBw.size();
    Mat Temp = Mat::zeros(m_Size.height + 2, m_Size.width + 2, srcBw.type());//延展图像
    srcBw.copyTo(Temp(Range(1, m_Size.height + 1), Range(1, m_Size.width + 1)));

    cv::floodFill(Temp, Point(0, 0), Scalar(255));//填充区域

    Mat cutImg;//裁剪延展的图像
    Temp(Range(1, m_Size.height + 1), Range(1, m_Size.width + 1)).copyTo(cutImg);

    dstBw = srcBw | (~cutImg);
}
bool isInside(Rect rect1, Rect rect2)
{
    Rect t = rect1&rect2;
    if (rect1.area() > rect2.area())
    {
        return false;
    }
    else
    {
        if (t.area() != 0)
            return true;
    }
}

int Card_Detector( )
{
        VideoCapture cap("/Users/machinelee/Desktop/1.mp4");
    if (!cap.isOpened())
    {
        cout<<"can't laod"<<endl;
        return -1;
    }

    int count=0;
    while (count < 540) {

        Mat srcImg;
        if (!cap.read(srcImg))
            break;
    resize(srcImg,srcImg,Size(srcImg.cols/2,srcImg.rows/2));
    int width = srcImg.cols;
    int height = srcImg.rows;
    int x, y;
    double B = 0.0, G = 0.0, R = 0.0, H = 0.0, S = 0.0, V = 0.0;
    Mat vec_rgb = Mat::zeros(srcImg.size(), CV_8UC1);
    for (x = 0; x < height; x++)
    {
        for (y = 0; y < width; y++)
        {
            B = srcImg.at<Vec3b>(x, y)[0];
            G = srcImg.at<Vec3b>(x, y)[1];
            R = srcImg.at<Vec3b>(x, y)[2];
            RGB2HSV(R, G, B, H, S, V);
            //红色范围，范围参考的网上。可以自己调
            if ((H >= 312 && H <= 360 || H >= 0 && H <= 20) && (S >= 17 && S <= 100) && (V>18 && V < 100))
                vec_rgb.at<uchar>(x, y) = 255;
            else if ((H >= 200 && H <= 248) && (S >= 17 && S <= 100) && (V>18 && V < 100))
                vec_rgb.at<uchar>(x, y) = 255;
//            else if ((H >= 52 && H <= 68) && (S >= 17 && S <= 100) && (V>18 && V < 100))
//                vec_rgb.at<uchar>(x, y) = 255;
            else
                continue;
        }
    }
//    imshow("hsv", vec_rgb);
    vector<vector<Point>>contours; //轮廓
    vector<Vec4i> hierarchy;//分层
    findContours(vec_rgb, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));//寻找轮廓
    vector<vector<Point>> contours_poly(contours.size());  //近似后的轮廓点集
    vector<Rect> boundRect(contours.size());  //包围点集的最小矩形vector
    vector<Point2f> center(contours.size());  //包围点集的最小圆形vector
    vector<float> radius(contours.size());  //包围点集的最小圆形半径vector

    Mat drawing = Mat::zeros(vec_rgb.size(), CV_8UC3);
    for (int i = 0; i< contours.size(); i++)
    {
                approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true); //对多边形曲线做适当近似，contours_poly[i]是输出的近似点集
                boundRect[i] = boundingRect(Mat(contours_poly[i])); //计算并返回包围轮廓点集的最小矩形
                minEnclosingCircle(contours_poly[i], center[i], radius[i]);//计算并返回包围轮廓点集的最小圆形及其半径
        Rect rect = boundRect[i];
                //首先进行一定的限制，筛选出区域
                //若轮廓矩形内部还包含着矩形，则将被包含的小矩形取消
                bool inside = false;
                for (int j = 0; j < contours.size(); j++)
                {
                    Rect t = boundRect[j];
                    if (rect == t)
                        continue;
                    else if (isInside(rect, t))
                    {
                        inside = true;
                        break;
                    }
                }

                if (inside)
                    continue;
        float ratio = (float)rect.width / (float)rect.height;
        if(contourArea(contours[i])<300||arcLength(contours[i],true)<40||ratio<=0.3||ratio>1.5){
            continue;
        }
        Scalar color = (0, 0, 255);//蓝色线画轮廓
//        drawContours(srcImg, contours_poly, i, color, 1, 8, hierarchy, 0, Point());//根据轮廓点集contours_poly和轮廓结构hierarchy画出轮廓
        rectangle(srcImg, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);//画矩形，tl矩形左上角，br右上角
//        circle(srcImg, center[i], (int)radius[i], color, 2, 8, 0);                                        //画圆形
    }
   imshow("src",srcImg);
   count += 1;
   if(waitKey(25)==27)
       break;
    }
    /// 显示在一个窗口
    return 0;
}

