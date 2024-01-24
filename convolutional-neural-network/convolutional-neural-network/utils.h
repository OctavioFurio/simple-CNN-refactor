#pragma once

#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <cassert>
#include <opencv2/opencv.hpp>







namespace Utils {
	Eigen::MatrixXd ImageToMatrix(cv::Mat mat);
	cv::Mat MatrixToImage(Eigen::MatrixXd matrix);
}




