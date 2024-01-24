#pragma once


#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <cassert>
#include <opencv2/opencv.hpp>
#include "utils.h"

using Matrix = Eigen::MatrixXd;



class ConvolutionLayer {

	private:
		Eigen::MatrixXd _kernel;
		int _padding;
		double _bias;

	public:
		ConvolutionLayer(Eigen::MatrixXd kernel = Eigen::MatrixXd::Ones(3,3), int padding = 0);
		ConvolutionLayer(std::vector<std::vector<double>> kernel, int padding = 0);
		~ConvolutionLayer();

		Eigen::MatrixXd CalculateConvolution(Eigen::MatrixXd& input);
		Eigen::MatrixXd CalculateConvolution(cv::Mat& image);

		static Eigen::MatrixXd Convolution2D(Eigen::MatrixXd& input, Eigen::MatrixXd& kernel);
		static Eigen::MatrixXd Convolution2D(Eigen::MatrixXd& input, Eigen::MatrixXd& kernel, int padding);

		Eigen::MatrixXd Kernel();

};


