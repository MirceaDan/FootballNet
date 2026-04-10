#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <fstream>

using namespace cv;
using namespace std;

const double BALL_DIAMETER_METERS = 0.22; 
const double FOCAL_LENGTH_PX = 600.0; 

int main() {
    VideoCapture cap("/home/mircea/Desktop/FootballNet/entrytrim/rgb.avi");
    if (!cap.isOpened()) return -1;

    int frame_width = cap.get(CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(CAP_PROP_FPS);
    if (fps <= 0) fps = 30;

    VideoWriter output("/home/mircea/Desktop/FootballNet/entrytrim/output_rgb.avi", VideoWriter::fourcc('M','J','P','G'), fps, Size(frame_width * 2, frame_height));

    // open csv fiel
    ofstream csvFile("/home/mircea/Desktop/FootballNet/entrytrim/trajectory.csv");
    csvFile << "Time(s),X(m),Y(m),Z(m),Distance(m)\n";

    // Kalman Filter Setup
    KalmanFilter KF(4, 2, 0);
    KF.transitionMatrix = (Mat_<float>(4, 4) << 1,0,1,0,   0,1,0,1,  0,0,1,0,  0,0,0,1);
    setIdentity(KF.measurementMatrix);
    setIdentity(KF.processNoiseCov, Scalar::all(1e-4));
    setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));
    setIdentity(KF.errorCovPost, Scalar::all(1));

    vector<Point2f> trajectory;
    Point2f last_valid_center(-1, -1);
    int frame_count = 0;

    while (true) {
        Mat frame, hsv, mask;
        cap >> frame;
        if (frame.empty()) break;

        // 1. Detection
        cvtColor(frame, hsv, COLOR_BGR2HSV);
        Mat mask1, mask2;
        inRange(hsv, Scalar(0, 120, 70), Scalar(10, 255, 255), mask1);
        inRange(hsv, Scalar(170, 120, 70), Scalar(180, 255, 255), mask2);
        mask = mask1 | mask2;
        GaussianBlur(mask, mask, Size(9, 9), 2);

        vector<vector<Point>> contours;
        findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        Point2f current_center;
        float current_radius = 0;
        bool found = false;

        sort(contours.begin(), contours.end(), [](const vector<Point>& a, const vector<Point>& b) {
            return contourArea(a) > contourArea(b);
        });

        for (auto& contour : contours) {
            double area = contourArea(contour);
            if (area > 400 && area < 15000) {
                minEnclosingCircle(contour, current_center, current_radius);
                double circularity = area / (CV_PI * current_radius * current_radius);

                if (circularity > 0.6) {
                    if (last_valid_center.x != -1 && norm(current_center - last_valid_center) > 150) continue;
                    found = true;
                    last_valid_center = current_center;
                    break; 
                }
            }
        }

        // 2. Kalman & 3D Logic
        KF.predict();
        double x_m = 0, y_m = 0, z_m = 0, dist_total = 0;

        if (found) {
            Mat measurement = (Mat_<float>(2, 1) << current_center.x, current_center.y);
            KF.correct(measurement);
            trajectory.push_back(current_center);

            // Compute 3D coordinates relative to the camera
            // Z (depth)
            z_m = (FOCAL_LENGTH_PX * BALL_DIAMETER_METERS) / (current_radius * 2.0);
            // X (horizontal) - offset from image center
            x_m = (current_center.x - frame_width / 2.0) * z_m / FOCAL_LENGTH_PX;
            // Y (vertical) - offset from image center
            y_m = (current_center.y - frame_height / 2.0) * z_m / FOCAL_LENGTH_PX;
            
            dist_total = sqrt(x_m*x_m + y_m*y_m + z_m*z_m);

            csvFile << (double)frame_count / fps << "," 
                    << x_m << "," << y_m << "," << z_m << "," 
                    << dist_total << "\n";
        }

        // 3. Left original iamge render
        Mat leftSide = frame.clone();
        if (found) {
            circle(leftSide, current_center, current_radius, Scalar(0, 255, 0), 2);
            string distText = "Z: " + to_string(z_m).substr(0, 4) + "m";
            putText(leftSide, distText, current_center + Point2f(20, 0), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2);
        }

        for (size_t i = 1; i < trajectory.size(); i++) {
            if (norm(trajectory[i] - trajectory[i-1]) < 100)
                line(leftSide, trajectory[i-1], trajectory[i], Scalar(0, 255, 255), 2, LINE_AA);
        }

        // 4. Right render (Top-View)
        Mat rightSide = Mat::zeros(frame_height, frame_width, CV_8UC3);
        for(int i=0; i<10; i++) line(rightSide, Point(0, i*50), Point(frame_width, i*50), Scalar(40,40,40));
        
        if (found) {
            int mapX = frame_width/2 + (x_m * 150); 
            int mapY = frame_height - (z_m * 50); 
            circle(rightSide, Point(mapX, mapY), 8, Scalar(0, 0, 255), -1);
        }
        putText(rightSide, "TOP VIEW (X-Z Floor Plan)", Point(20, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255,255,255), 2);

        // final bindings
        Mat combined(frame_height, frame_width * 2, CV_8UC3);
        leftSide.copyTo(combined(Rect(0, 0, frame_width, frame_height)));
        rightSide.copyTo(combined(Rect(frame_width, 0, frame_width, frame_height)));

        output.write(combined);
        frame_count++;
    }

    csvFile.close();
    cap.release();
    output.release();
    return 0;
}