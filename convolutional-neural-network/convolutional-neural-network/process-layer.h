#pragma once

#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <cassert>
#include <opencv2/opencv.hpp>
#include "convolution-layer.h"
#include "activation-function.h"
#include "pooling-operation.h"
#include "utils.h"

using Matrix = Eigen::MatrixXd;



class ProcessLayer {

	private:
		Eigen::MatrixXd _convolutionOutput;
		Eigen::MatrixXd _poolingOutput;

		ConvolutionLayer _convolutionLayer;
		IActivationFunction* _activationFunction;
		IPooling* _poolingLayer;

	public:
		ProcessLayer(Eigen::MatrixXd& kernel, int padding = 0, IActivationFunction* actFunc = new ReLU(), IPooling* pooling = new DontPooling());
		ProcessLayer(std::vector<std::vector<double>>& kernel, int padding = 0, IActivationFunction* actFunc = new ReLU(), IPooling* pooling = new DontPooling());
		~ProcessLayer();

		Eigen::MatrixXd CalculateConvolution(Eigen::MatrixXd input);
		Eigen::MatrixXd CalculateConvolution(cv::Mat image);

		Eigen::MatrixXd ApplayActivationFunction(Eigen::MatrixXd input);

		Eigen::MatrixXd CalculatePooling(Eigen::MatrixXd input);

		// std::vector<double> Flattening(Eigen::MatrixXd input);
		// Eigen::MatrixXd Reshape(std::vector<double> gradients, size_t rows, size_t cols);

		Eigen::MatrixXd ConvolutionOutout();
		Eigen::MatrixXd PoolingOutput();

		
		int KernelRows();
		int KernelCols();

		int PoolingRows();
		int PoolingCols();


};


