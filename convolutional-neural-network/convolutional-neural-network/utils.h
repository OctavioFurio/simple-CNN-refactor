#pragma once

#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <cassert>
#include <numeric>
#include <opencv2/opencv.hpp>







namespace Utils {

	Eigen::MatrixXd ImageToMatrix(cv::Mat mat);
	cv::Mat MatrixToImage(Eigen::MatrixXd matrix);


	std::vector<double> NormalizeVector(const std::vector<double> originalVector, double min, double max);


	double Mean(std::vector<double> values);
	double Variance(std::vector<double> values, double mean);
	std::vector<double> BatchNormalization(std::vector<double> originalInput);


	Eigen::MatrixXd FlipMatrixBy180Degree(Eigen::MatrixXd matrix);


	void StoreMatrixByForce(Eigen::MatrixXd* a, Eigen::MatrixXd b);

}




