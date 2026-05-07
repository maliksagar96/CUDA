#include <iostream>
#include <opencv2/opencv.hpp>

int main() {

  cv::Mat img = cv::imread("Lenna_test_image.png");

  if(img.empty()) {
    std::cout << "Image not loaded\n";
    return 1;
  }

  std::cout << "Width  : " << img.cols << '\n';
  std::cout << "Height : " << img.rows << '\n';

  return 0;
}