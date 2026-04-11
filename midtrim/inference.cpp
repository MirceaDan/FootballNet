#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>
#include <fstream>

using namespace cv;
using namespace cv::dnn;
using namespace std;

// Parametri pentru detecție și proiecție
const float CONFIDENCE_THRESHOLD = 0.5;
const double BALL_DIAMETER_METERS = 0.22; 
const double FOCAL_LENGTH_PX = 600.0;

int main() {
    // 1. Load YOLOv11 (ONNX)
    Net net = readNetFromONNX("/home/mircea/Desktop/FootballNet/midtrim/yolo11n.onnx");
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    VideoCapture cap("/home/mircea/Desktop/FootballNet/midtrim/rgb.avi");
    if (!cap.isOpened()) return -1;

    int frame_width = cap.get(CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(CAP_PROP_FPS);
    if (fps <= 0) fps = 30;

    VideoWriter output("/home/mircea/Desktop/FootballNet/midtrim/output_rgb.avi", 
                       VideoWriter::fourcc('M','J','P','G'), fps, Size(frame_width * 2, frame_height));

    ofstream csvFile("/home/mircea/Desktop/FootballNet/midtrim/trajectory.csv");
    csvFile << "Time(s),X(m),Y(m),Z(m),Total_Distance(m)\n";

    // 2. Setup Kalman Filter
    KalmanFilter KF(4, 2, 0);
    KF.transitionMatrix = (Mat_<float>(4, 4) << 1,0,1,0,   0,1,0,1,  0,0,1,0,  0,0,0,1);
    setIdentity(KF.measurementMatrix);
    setIdentity(KF.processNoiseCov, Scalar::all(1e-4));
    setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));
    setIdentity(KF.errorCovPost, Scalar::all(1));

    // Folosim Point2f pentru precizie
    vector<Point2f> trajectory; 
    int frame_count = 0;

    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        // YOLO Inference
        Mat blob = blobFromImage(frame, 1/255.0, Size(640, 640), Scalar(), true, false);
        net.setInput(blob);
        
        vector<Mat> outputs;
        net.forward(outputs, net.getUnconnectedOutLayersNames());

        Mat output_data = outputs[0].reshape(1, outputs[0].size[1]);
        transpose(output_data, output_data);

        float x_factor = (float)frame.cols / 640.0;
        float y_factor = (float)frame.rows / 640.0;

        Point2f ball_center;
        float ball_radius = 0;
        bool ball_found = false;

        for (int i = 0; i < output_data.rows; ++i) {
            float ball_score = output_data.at<float>(i, 36); 
            if (ball_score > CONFIDENCE_THRESHOLD) {
                float x = output_data.at<float>(i, 0) * x_factor;
                float y = output_data.at<float>(i, 1) * y_factor;
                float w = output_data.at<float>(i, 2) * x_factor;
                float h = output_data.at<float>(i, 3) * y_factor;

                ball_center = Point2f(x, y);
                ball_radius = (w + h) / 4.0;
                ball_found = true;
                break; 
            }
        }

        // 3. Kalman & 3D Logic
        KF.predict();
        double pos_x = 0, pos_y = 0, dist_z = 0, total_dist = 0;

        if (ball_found) {
            Mat measurement = (Mat_<float>(2, 1) << ball_center.x, ball_center.y);
            KF.correct(measurement);
            trajectory.push_back(ball_center);

            dist_z = (FOCAL_LENGTH_PX * BALL_DIAMETER_METERS) / (ball_radius * 2.0);
            pos_x  = (ball_center.x - frame_width/2.0) * dist_z / FOCAL_LENGTH_PX;
            pos_y  = (ball_center.y - frame_height/2.0) * dist_z / FOCAL_LENGTH_PX;
            total_dist = sqrt(pos_x*pos_x + pos_y*pos_y + dist_z*dist_z);

            csvFile << (double)frame_count / fps << "," 
                    << pos_x << "," << pos_y << "," 
                    << dist_z << "," << total_dist << "\n";
            csvFile.flush();
        }

        // --- 4. Randare vizuală (CORECȚIE BUG TRACKING) ---
        Mat canvas = Mat::zeros(frame_height, frame_width * 2, CV_8UC3);
        Mat left = frame.clone();
        
        if (ball_found) {
            circle(left, ball_center, ball_radius, Scalar(0, 255, 0), 2);
            putText(left, "Dist: " + to_string(total_dist).substr(0,4) + "m", 
                    Point(ball_center.x + 10, ball_center.y), 1, 1.5, Scalar(255,255,255), 2);
        }

        // DESENARE TRAIECTORIE FILTRATĂ
        // Începem desenul doar de la al doilea punct din vector
        for (size_t i = 1; i < trajectory.size(); i++) {
            // Verificăm distanța între punctul curent și cel anterior.
            double d = norm(trajectory[i] - trajectory[i-1]);

            // Daca distanta este prea mare (ex: peste 120px într-un cadru), refuzăm să unim punctele.
            // Această valoare poate fi ajustată. Dacă traiectoria se rupe prea des la mișcări rapide, mărește valoarea.
            if (d < 120.0) {
                // LINE_AA (Anti-Aliased) face linia mult mai fină, eliminând efectul de pixelare "în trepte" din imaginea ta.
                line(left, trajectory[i-1], trajectory[i], Scalar(0, 255, 255), 3, LINE_AA);
            }
        }

        Mat right = Mat::zeros(frame_height, frame_width, CV_8UC3);
        if (ball_found) {
            int mapX = frame_width/2 + (pos_x * 150); 
            int mapY = frame_height - (dist_z * 40);
            if (mapX >= 0 && mapX < frame_width && mapY >= 0 && mapY < frame_height)
                circle(right, Point(mapX, mapY), 8, Scalar(0, 0, 255), -1);
        }
        putText(right, "TOP VIEW (X-Z Plan)", Point(20, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255,255,255), 2);

        left.copyTo(canvas(Rect(0, 0, frame_width, frame_height)));
        right.copyTo(canvas(Rect(frame_width, 0, frame_width, frame_height)));

        output.write(canvas);
        frame_count++;
    }

    csvFile.close();
    cap.release();
    output.release();
    return 0;
}