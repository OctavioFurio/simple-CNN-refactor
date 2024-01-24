#pragma once

#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <cassert>
#include <opencv2/opencv.hpp>
#include "process-layer.h"
#include "multy-layer-perceptron.h"



struct MlpArchitecture {
	std::vector<size_t> architecture;
	IActivationFunction* activationFunction;
	double lerningRate;

	MlpArchitecture(std::vector<size_t> arch, IActivationFunction* actFun = new Tanh(), double _lerningRate = 0.03)
		: architecture(arch), activationFunction(actFun), lerningRate(_lerningRate) 
	{ }
};



class CNN {

	private:
		std::vector<ProcessLayer> _processLayers;
		MLP _mlp;
		double _learningRate;



	public:
		CNN(
			int inputRows
			, int inputCols
			, std::vector<ProcessLayer> processLayers
			, std::vector<size_t> mlpArchitecture
			, IActivationFunction* activationFunction = new Tanh()
			, double learningRate = 0.03
		);
		
		~CNN();

		std::vector<double> Forward(Eigen::MatrixXd input);



		std::vector<double> Flattening(Eigen::MatrixXd input);
		Eigen::MatrixXd Reshape(std::vector<double> gradients, size_t rows, size_t cols);

};


