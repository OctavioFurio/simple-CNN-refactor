#pragma once

#include "activation-function.h"
#include "convolution-layer.h"
#include "pooling-operation.h"
#include "utils.h"
#include <cassert>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

using Matrix = Eigen::MatrixXd;

class ProcessLayer {

private:
	ConvolutionLayer _convolutionLayer;
	IActivationFunction* _activationFunction;
	IPooling* _poolingLayer;

public:

	Eigen::MatrixXd _convolutionOutput;
	Eigen::MatrixXd _poolingOutput;

	ProcessLayer(Eigen::MatrixXd kernel, int padding = 0, IActivationFunction* actFunc = new ReLU(), IPooling* pooling = new DontPooling());
	ProcessLayer(std::vector<std::vector<double>>& kernel, int padding = 0, IActivationFunction* actFunc = new ReLU(), IPooling* pooling = new DontPooling());
	~ProcessLayer();

	Eigen::MatrixXd CalculateConvolution(Eigen::MatrixXd input);
	Eigen::MatrixXd CalculateConvolution(cv::Mat image);
	Eigen::MatrixXd ConvolutionBackward(Eigen::MatrixXd input, Eigen::MatrixXd incomeGradient, double learningRate = 0.003);

	Eigen::MatrixXd ApplayActivationFunction(Eigen::MatrixXd input);

	Eigen::MatrixXd CalculatePooling(Eigen::MatrixXd input);
	Eigen::MatrixXd PoolingBackward(Eigen::MatrixXd input);

	Eigen::MatrixXd ConvolutionOutout();
	Eigen::MatrixXd PoolingOutput();
	Eigen::MatrixXd Output();

	int KernelRows();
	int KernelCols();

	int PoolingRows();
	int PoolingCols();

	Eigen::MatrixXd Kernel();

	Eigen::MatrixXd Gradient();

	void NormalizeKernel();
};