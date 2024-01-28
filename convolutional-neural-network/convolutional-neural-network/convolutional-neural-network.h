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




class CNN {

	private:
		std::vector<ProcessLayer> _processLayers;
		MLP _mlp;
		double _learningRate;

		std::vector<double> _flattedMatrix;
		size_t _reshapeRows;
		size_t _reshapeCols;


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
		std::vector<double> Backward(std::vector<double> correctOutputs, Eigen::MatrixXd input);


		static std::vector<double> Flattening(Eigen::MatrixXd input);
		Eigen::MatrixXd Reshape(std::vector<double> gradients, size_t rows, size_t cols);

		friend std::ostream& operator<<(std::ostream& os, CNN cnn);

};


