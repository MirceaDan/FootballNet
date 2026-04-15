#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <iostream>

int main() {

    // ================= CONFIG =================
    const std::string VIDEO_PATH = "rgb.avi";
    const std::string MODEL_PATH = "moca_bg_det.pt";

    const int IMG_SIZE = 320;

    // ================= LOAD MODEL =================
    torch::jit::Module model = torch::jit::load(MODEL_PATH);
    model.to(torch::kCPU);
    model.eval();

    // ================= VIDEO =================
    cv::VideoCapture cap(VIDEO_PATH);
    if (!cap.isOpened()) {
        std::cout << "Error opening video\n";
        return -1;
    }

    int w = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int h = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    // ================= FRAME BUFFERS =================
    cv::Mat frame, prev_frame;

    // read first frame
    if (!cap.read(prev_frame)) {
        std::cout << "Cannot read first frame\n";
        return -1;
    }

    int frame_id = 1;

    while (cap.read(frame)) {

        // ================= PREPROCESS =================
        auto preprocess = [&](cv::Mat& img) {
            cv::Mat resized;
            cv::resize(img, resized, cv::Size(IMG_SIZE, IMG_SIZE));
            cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
            resized.convertTo(resized, CV_32F, 1.0 / 255.0);

            torch::Tensor t = torch::from_blob(
                resized.data,
                {1, IMG_SIZE, IMG_SIZE, 3},
                torch::kFloat32
            );

            t = t.permute({0, 3, 1, 2}).clone(); // NHWC -> NCHW
            return t;
        };

        torch::Tensor t_cur = preprocess(frame);
        torch::Tensor t_prev = preprocess(prev_frame);

        // ================= INFERENCE =================
        torch::NoGradGuard no_grad;

        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(t_cur);
        inputs.push_back(t_prev);

        auto output = model.forward(inputs).toTuple();

        // unpack outputs: (P_hat, M, xyz_pred)
        torch::Tensor P_hat = output->elements()[0].toTensor(); // [1,1,H,W]
        torch::Tensor M     = output->elements()[1].toTensor();
        torch::Tensor xyz   = output->elements()[2].toTensor(); // [1,3]

        float X = xyz[0][0].item<float>();
        float Y = xyz[0][1].item<float>();
        float Z = xyz[0][2].item<float>();

        // ================= PROJECT TO IMAGE =================
        float f = 600.0f;
        int cx = IMG_SIZE / 2;
        int cy = IMG_SIZE / 2;

        int u = (int)((X * f) / (Z + 1e-6) + cx);
        int v = (int)((Y * f) / (Z + 1e-6) + cy);

        // scale back to original resolution
        u = u * w / IMG_SIZE;
        v = v * h / IMG_SIZE;

        // ================= VISUALIZE =================
        cv::circle(frame, cv::Point(u, v), 8, cv::Scalar(0, 255, 0), -1);

        std::string text = "Z=" + std::to_string(Z).substr(0, 5);
        cv::putText(frame, text, cv::Point(u + 10, v),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0,255,0), 1);

        // ================= OPTIONAL: SHOW HEATMAP =================
        torch::Tensor P = P_hat.squeeze().detach().cpu();
        P = P / (P.max() + 1e-6);

        cv::Mat heatmap(P.size(0), P.size(1), CV_32F, P.data_ptr<float>());
        cv::resize(heatmap, heatmap, cv::Size(w, h));

        cv::Mat heatmap_u8;
        heatmap.convertTo(heatmap_u8, CV_8U, 255);

        cv::applyColorMap(heatmap_u8, heatmap_u8, cv::COLORMAP_JET);

        cv::Mat overlay;
        cv::addWeighted(frame, 0.7, heatmap_u8, 0.3, 0, overlay);

        cv::imshow("MoCA-BG-DETR Output", overlay);

        if (cv::waitKey(1) == 27) break;

        // ================= UPDATE =================
        prev_frame = frame.clone();
        frame_id++;
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}