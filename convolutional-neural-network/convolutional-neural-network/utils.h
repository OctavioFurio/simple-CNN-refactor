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

	std::vector<std::string> SplitString(const std::string& input, const std::string& delimiter);

	Eigen::MatrixXd ReshapeMatrix(std::vector<double> gradients, size_t rows, size_t cols);
	std::vector<double> FlatMatrix(Eigen::MatrixXd input);
	
	Eigen::MatrixXd BatchNormalization(Eigen::MatrixXd mat);
}




