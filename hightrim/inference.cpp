#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>

int main() {
    // ================= 1. CONFIG =================
    const std::string VIDEO_PATH  = "/home/mircea/Desktop/FootballNet/hightrim/rgb.avi";
    const std::string MODEL_PATH  = "/home/mircea/Desktop/FootballNet/hightrim/moca_bg_det.pth";
    const std::string OUTPUT_PATH = "/home/mircea/Desktop/FootballNet/hightrim/output_moca.avi";
    const std::string CSV_PATH    = "/home/mircea/Desktop/FootballNet/hightrim/trajectory.csv";

    const int IMG_SIZE = 320;
    const float HEATMAP_THRESHOLD = 0.3f;

    // ================= 2. LOAD MODEL =================
    torch::jit::Module model;
    try {
        model = torch::jit::load(MODEL_PATH);
        model.to(torch::kCPU);
        model.eval();
    } catch (const c10::Error& e) {
        std::cerr << "Error loading model\n";
        return -1;
    }

    // ================= 3. KALMAN FILTER SETUP =================
    // Smoothen 3D coordinates (X, Y, Z)
    cv::KalmanFilter kf(6, 3, 0);
    kf.transitionMatrix = (cv::Mat_<float>(6, 6) << 
        1,0,0,1,0,0,   0,1,0,0,1,0,   0,0,1,0,0,1, 
        0,0,0,1,0,0,   0,0,0,0,1,0,   0,0,0,0,0,1);
    
    cv::setIdentity(kf.measurementMatrix);
    cv::setIdentity(kf.processNoiseCov, cv::Scalar::all(1e-4));
    cv::setIdentity(kf.measurementNoiseCov, cv::Scalar::all(1e-1));
    cv::setIdentity(kf.errorCovPost, cv::Scalar::all(1));

    // ================= 4. VIDEO & CSV =================
    cv::VideoCapture cap(VIDEO_PATH);
    int w = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int h = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = 30;

    cv::VideoWriter writer(OUTPUT_PATH, cv::VideoWriter::fourcc('M','J','P','G'), fps, cv::Size(w * 2, h));
    std::ofstream csvFile(CSV_PATH);
    csvFile << "Time(s),X(m),Y(m),Z(m),Distance(m)\n";

    std::vector<cv::Point> trajectory_2d;
    cv::Mat prev_frame;
    cap >> prev_frame;
    cv::resize(prev_frame, prev_frame, cv::Size(IMG_SIZE, IMG_SIZE));

    int frame_id = 0;
    bool kf_initialized = false;

    while (true) {
        cv::Mat frame;
        if (!cap.read(frame)) break;

        cv::Mat cur_resized;
        cv::resize(frame, cur_resized, cv::Size(IMG_SIZE, IMG_SIZE));

        // Preprocess
        auto to_tensor = [&](cv::Mat& img) {
            cv::Mat rgb;
            cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
            rgb.convertTo(rgb, CV_32F, 1.0/255.0);
            return torch::from_blob(rgb.data, {1, IMG_SIZE, IMG_SIZE, 3}, torch::kFloat32).permute({0,3,1,2}).clone();
        };

        torch::Tensor t_cur = to_tensor(cur_resized);
        torch::Tensor t_prev = to_tensor(prev_frame);

        // Inference
        torch::NoGradGuard no_grad;
        auto output = model.forward({t_cur, t_prev}).toTuple();
        torch::Tensor P_hat = output->elements()[0].toTensor().squeeze().detach(); 
        torch::Tensor xyz_pred = output->elements()[2].toTensor().detach();

        // Argmax simplu pe Heatmap
        float max_val = P_hat.max().item<float>();
        int max_idx = P_hat.argmax().item<int>();
        int H_feat = P_hat.size(0), W_feat = P_hat.size(1);
        int y_idx = max_idx / W_feat;
        int x_idx = max_idx % W_feat;

        cv::Mat left_view = frame.clone();
        cv::Mat right_view = cv::Mat::zeros(h, w, CV_8UC3);

        if (max_val > HEATMAP_THRESHOLD) {
            float x_raw = xyz_pred[0][0].item<float>();
            float y_raw = xyz_pred[0][1].item<float>();
            float z_raw = xyz_pred[0][2].item<float>();

            // Kalman filter update
            if (!kf_initialized) {
                kf.statePost.at<float>(0) = x_raw; kf.statePost.at<float>(1) = y_raw; kf.statePost.at<float>(2) = z_raw;
                kf_initialized = true;
            }
            kf.predict();
            cv::Mat meas = (cv::Mat_<float>(3,1) << x_raw, y_raw, z_raw);
            cv::Mat est = kf.correct(meas);

            float xf = est.at<float>(0), yf = est.at<float>(1), zf = est.at<float>(2);
            float dist = std::sqrt(xf*xf + yf*yf + zf*zf);

            // 1. Write 2 CSV
            csvFile << std::fixed << std::setprecision(3) << (frame_id/fps) << "," << xf << "," << yf << "," << zf << "," << dist << "\n";

            // 2. 2D Traiectory (Overlay Video)
            int cx = (int)(x_idx * (w / (float)W_feat));
            int cy = (int)(y_idx * (h / (float)H_feat));
            trajectory_2d.push_back(cv::Point(cx, cy));

            cv::circle(left_view, cv::Point(cx, cy), 12, cv::Scalar(0, 255, 0), 2);
            cv::putText(left_view, "D:" + std::to_string(dist).substr(0,4) + "m", cv::Point(cx+10, cy-10), 1, 1.2, cv::Scalar(0, 255, 0), 2);

            // 3. Top-View
            float scale_x = w / 6.0f; float scale_z = h / 10.0f;
            int tx = (int)(w/2 + xf * scale_x);
            int tz = (int)(h - zf * scale_z);
            cv::circle(right_view, cv::Point(tx, tz), 8, cv::Scalar(0, 0, 255), -1);
        }

        // Draw trajectory to original video
        for (size_t i = 1; i < trajectory_2d.size(); i++) {
            if(cv::norm(trajectory_2d[i] - trajectory_2d[i-1]) < 100)
                cv::line(left_view, trajectory_2d[i-1], trajectory_2d[i], cv::Scalar(0, 255, 255), 2);
        }

        // Labeling
        cv::circle(right_view, cv::Point(w/2, h-15), 6, cv::Scalar(255, 0, 0), -1); // Camera
        cv::putText(right_view, "TOP-VIEW MAP (X-Z)", cv::Point(20, 30), 1, 1.5, cv::Scalar(255, 255, 255), 2);

        // Final Canvas
        cv::Mat canvas(h, w * 2, CV_8UC3);
        left_view.copyTo(canvas(cv::Rect(0, 0, w, h)));
        right_view.copyTo(canvas(cv::Rect(w, 0, w, h)));

        writer.write(canvas);
        prev_frame = cur_resized.clone();
        frame_id++;
    }

    cap.release(); writer.release(); csvFile.close();
    return 0;
}