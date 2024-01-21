#pragma once

#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include "activation-function.h"
#include "pooling-operation.h"

using Matrix = Eigen::MatrixXd;



class ProcessLayer {

	private:
		Eigen::MatrixXd _kernel;
		//Eigen::MatrixXd _poolingMatrix;
		IActivationFunction* _activationFunction;
		IPooling* _poolingOperation;
		int _padding;
		double _bias;


	public:
		ProcessLayer(Eigen::MatrixXd& kernel, int padding, IActivationFunction* actFunc = new ReLU(), IPooling* pooling = new DontPooling());
		~ProcessLayer();

		Eigen::MatrixXd CalculateConvolution(Eigen::MatrixXd input);
		Eigen::MatrixXd CalculateConvolution(cv::Mat image);
		
		Eigen::MatrixXd ApplayActivationFunction(Eigen::MatrixXd input);

		Eigen::MatrixXd CalculatePooling(Eigen::MatrixXd input);
		
		std::vector<double> Flattening(Eigen::MatrixXd input);


		// private
		static Eigen::MatrixXd Convolution2D(Eigen::MatrixXd input, Eigen::MatrixXd kernel);
		static Eigen::MatrixXd Convolution2D(Eigen::MatrixXd input, Eigen::MatrixXd kernel, int padding);
		static Eigen::MatrixXd ImageToMatrix(cv::Mat mat);
		static cv::Mat MatrixToImage(Eigen::MatrixXd matrix);
		
 
};


