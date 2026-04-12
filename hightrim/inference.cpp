#include <fstream>
#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <torch/torch.h>
#include <vector>

int main() {

    // ================= CONFIG =================
    /*const std::string INPUT_VIDEO = "rgb.avi";
    const std::string OUTPUT_VIDEO = "output_rgb.avi";
    const std::string OUTPUT_CSV = "trajectory.csv";
    const std::string MODEL_PATH = "ball_yolo.pth";*/

    const std::string INPUT_VIDEO = "/home/mircea/Desktop/FootballNet/midtrim/rgb.avi";
    const std::string OUTPUT_VIDEO = "/home/mircea/Desktop/FootballNet/midtrim/output_rgb.avi";
    const std::string OUTPUT_CSV = "/home/mircea/Desktop/FootballNet/midtrim/trajectory.csv";
    const std::string MODEL_PATH = "/home/mircea/Desktop/FootballNet/midtrim/ball_yolo.pth";

    const float CONF_THRESHOLD = 0.5f;

    const float FOCAL_LENGTH = 800.0f;
    const float BALL_DIAMETER = 0.22f;

    // ================= LOAD MODEL =================
    torch::jit::Module model = torch::jit::load(MODEL_PATH);
    model.to(torch::kCPU);
    model.eval();

    // ================= VIDEO =================
    cv::VideoCapture cap(INPUT_VIDEO);
    if (!cap.isOpened()) {
        printf("Error opening video\n");
        return -1;
    }

    int w = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int h = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int fps = (int)cap.get(cv::CAP_PROP_FPS);

    int cx = w / 2;
    int cy = h / 2;

    cv::VideoWriter out;
    out.open(OUTPUT_VIDEO,
             cv::VideoWriter::fourcc('X','V','I','D'),
             fps,
             cv::Size(w * 2, h));

    // ================= CSV =================
    std::ofstream csv(OUTPUT_CSV);
    csv << "frame,X,Y,Z\n";

    // ================= KALMAN =================
    cv::KalmanFilter kf(6, 3);

    kf.measurementMatrix = cv::Mat::eye(3, 6, CV_32F);

    kf.transitionMatrix = (cv::Mat_<float>(6,6) <<
        1,0,0,1,0,0,
        0,1,0,0,1,0,
        0,0,1,0,0,1,
        0,0,0,1,0,0,
        0,0,0,0,1,0,
        0,0,0,0,0,1
    );

    kf.processNoiseCov = cv::Mat::eye(6,6,CV_32F) * 1e-2;
    kf.measurementNoiseCov = cv::Mat::eye(3,3,CV_32F) * 1e-1;

    // ================= TRAJECTORIES =================
    std::vector<cv::Point> traj_2d;
    std::vector<cv::Point2f> traj_top;

    cv::Mat frame;
    int frame_id = 0;

    while (cap.read(frame)) {

        // ================= PREPROCESS =================
        cv::Mat resized;
        cv::resize(frame, resized, cv::Size(320, 320));

        cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
        resized.convertTo(resized, CV_32F, 1.0 / 255.0);

        torch::Tensor input_tensor = torch::from_blob(
            resized.data,
            {1, 320, 320, 3},
            torch::kFloat32
        );

        input_tensor = input_tensor.permute({0, 3, 1, 2}); // NHWC -> NCHW
        input_tensor = input_tensor.clone(); // IMPORTANT (avoid memory issues)

        // ================= INFERENCE =================
        torch::NoGradGuard no_grad;

        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);

        at::Tensor output = model.forward(inputs).toTensor();

        float* data = output.data_ptr<float>();

        float xc = data[0];
        float yc = data[1];
        float bw = data[2];
        float bh = data[3];
        float conf = data[4];

        if (conf > CONF_THRESHOLD) {
            int input_size = 320;

            float x1_320 = (xc - bw/2) * input_size;
            float y1_320 = (yc - bh/2) * input_size;
            float x2_320 = (xc + bw/2) * input_size;
            float y2_320 = (yc + bh/2) * input_size;

            float scale_x = (float)w / input_size;
            float scale_y = (float)h / input_size;

            int x1 = (int)(x1_320 * scale_x);
            int y1 = (int)(y1_320 * scale_y);
            int x2 = (int)(x2_320 * scale_x);
            int y2 = (int)(y2_320 * scale_y);

            int u = (x1 + x2) / 2;
            int v = (y1 + y2) / 2;

            int d = std::max(x2 - x1, y2 - y1);

            float X=0, Y=0, Z=0;

            if (d > 0) {

                // ================= DEPTH =================
                Z = (FOCAL_LENGTH * BALL_DIAMETER) / d;

                // ================= 3D =================
                X = (u - cx) * Z / FOCAL_LENGTH;
                Y = (v - cy) * Z / FOCAL_LENGTH;

                cv::Mat measurement = (cv::Mat_<float>(3,1) << X, Y, Z);
                kf.correct(measurement);

                cv::Mat pred = kf.predict();

                float Xp = pred.at<float>(0);
                float Yp = pred.at<float>(1);
                float Zp = pred.at<float>(2);

                // ================= SAVE =================
                csv << frame_id << "," << Xp << "," << Yp << "," << Zp << "\n";

                // ================= DRAW =================
                int radius = d / 2;

                cv::circle(frame, cv::Point(u,v), radius, cv::Scalar(0,255,0), 2);
                cv::circle(frame, cv::Point(u,v), 4, cv::Scalar(0,0,255), -1);

                cv::putText(frame,
                    "Z=" + std::to_string(Zp),
                    cv::Point(x1, y1-10),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.5,
                    cv::Scalar(0,255,0),
                    1
                );

                traj_2d.push_back(cv::Point(u,v));
                traj_top.push_back(cv::Point2f(Xp, Zp));
            }

        } else {
            kf.predict();
        }

        // ================= DRAW 2D TRAJECTORY =================
        for (size_t i = 1; i < traj_2d.size(); i++) {
            cv::line(frame, traj_2d[i-1], traj_2d[i],
                     cv::Scalar(255,0,0), 2);
        }

        // ================= TOP VIEW =================
        cv::Mat map = cv::Mat::zeros(h, w, CV_8UC3);

        float scale = 200.0f;

        for (size_t i = 1; i < traj_top.size(); i++) {

            float x1 = traj_top[i-1].x;
            float z1 = traj_top[i-1].y;

            float x2 = traj_top[i].x;
            float z2 = traj_top[i].y;

            cv::Point p1(w/2 + x1 * scale, h - z1 * scale);
            cv::Point p2(w/2 + x2 * scale, h - z2 * scale);

            cv::line(map, p1, p2, cv::Scalar(0,255,255), 2);
        }

        // ================= COMBINE =================
        cv::Mat combined;
        cv::hconcat(frame, map, combined);

        out.write(combined);

        frame_id++;
    }

    // ================= CLEANUP =================
    cap.release();
    out.release();
    csv.close();
    cv::destroyAllWindows();

    return 0;
}