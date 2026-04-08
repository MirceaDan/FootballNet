#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

int main() {
    cv::dnn::Net net = cv::dnn::readNetFromONNX("ball_yolo.onnx");

    cv::VideoCapture cap("input.avi");

    int w = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int h = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    cv::Mat frame;

    while (cap.read(frame)) {
        cv::Mat blob;
        cv::dnn::blobFromImage(frame, blob, 1/255.0, cv::Size(224,224));

        net.setInput(blob);
        cv::Mat out = net.forward();

        float* data = (float*)out.data;

        float xc = data[0];
        float yc = data[1];
        float bw = data[2];
        float bh = data[3];
        float conf = data[4];

        if (conf > 0.5) {
            int x = (xc - bw/2) * w;
            int y = (yc - bh/2) * h;
            int ww = bw * w;
            int hh = bh * h;

            cv::rectangle(frame, cv::Rect(x,y,ww,hh), cv::Scalar(0,255,0), 2);
        }

        cv::imshow("out", frame);
        if (cv::waitKey(1) == 27) break;
    }
}