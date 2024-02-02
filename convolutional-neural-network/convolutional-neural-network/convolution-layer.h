#pragma once


#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <cassert>
#include <opencv2/opencv.hpp>
#include "kernels.h"
#include "utils.h"

using Matrix = Eigen::MatrixXd;



class ConvolutionLayer {

	private:
		Eigen::MatrixXd _kernel;
		Eigen::MatrixXd _gradient;
		int _padding;
		double _bias;

	public:
		ConvolutionLayer(Eigen::MatrixXd kernel = Eigen::MatrixXd::Ones(3,3), int padding = 0);
		ConvolutionLayer(std::vector<std::vector<double>> kernel, int padding = 0);
		~ConvolutionLayer();

		Eigen::MatrixXd CalculateConvolution(Eigen::MatrixXd input);
		Eigen::MatrixXd CalculateConvolution(cv::Mat image);
		Eigen::MatrixXd Backward(Eigen::MatrixXd input, Eigen::MatrixXd incomeGradient, double learningRate = 0.03);

		static Eigen::MatrixXd Convolution2D(Eigen::MatrixXd input, Eigen::MatrixXd kernel);
		static Eigen::MatrixXd Convolution2D(Eigen::MatrixXd input, Eigen::MatrixXd kernel, int padding);
		static Eigen::MatrixXd& Convolution2D(Eigen::MatrixXd input, Eigen::MatrixXd kernel, int rowPadding, int colParing);
		static Eigen::MatrixXd Convolution2DWithVisualization(Eigen::MatrixXd& input, Eigen::MatrixXd& kernel);

		Eigen::MatrixXd Kernel();

		Eigen::MatrixXd Gradient();

		void NormalizeKernel();

};


