#include "convolutional-neural-network.h"





CNN::CNN(
	int inputRows
	, int inputCols
	, std::vector<ProcessLayer> processLayers
	, std::vector<size_t> mlpArchitecture
	, IActivationFunction* activationFunction
	, double learningRate
) {

	_processLayers = processLayers;
	_learningRate = learningRate;

	int rows = inputRows;
	int cols = inputCols;

	for (auto processLayer : processLayers) {
		rows  =  rows - processLayer.KernelRows() - processLayer.PoolingRows() + 2;
		cols  =  cols - processLayer.KernelCols() - processLayer.PoolingCols() + 2;

		assert( rows > 0   &&   cols > 0 );
	}


	size_t mlpInputSize  =  rows * cols;

	_mlp  =  MLP(mlpArchitecture, mlpInputSize, activationFunction, learningRate);
}



CNN::~CNN()
{
}



std::vector<double> CNN::Forward(Eigen::MatrixXd input)
{
	Eigen::MatrixXd processResult = input;


	for (auto processLayer : _processLayers) {
		 Eigen::MatrixXd convolvedMatrix  =  processLayer.CalculateConvolution(processResult);
		 Eigen::MatrixXd activatedMatrix  =  processLayer.ApplayActivationFunction(convolvedMatrix);
		 Eigen::MatrixXd pooledMatrix	 =  processLayer.CalculatePooling( activatedMatrix );

		 processResult  =  pooledMatrix;
	}

	std::vector<double> flattedMatrix  =  Flattening( processResult );

	std::vector<double> mlpOutput  =  _mlp.Forward( flattedMatrix );

	return mlpOutput;
}



std::vector<double> CNN::Flattening(Eigen::MatrixXd input)
{
	std::vector<double> vec;
	vec.reserve(input.size());

	for (int i = 0; i < input.rows(); i++) {
		for (int j = 0; j < input.cols(); j++) {
			vec.push_back(input(i, j));
		}
	}

	return vec;
}



Eigen::MatrixXd CNN::Reshape(std::vector<double> gradients, size_t rows, size_t cols)
{
	Eigen::MatrixXd reshapedMatrix  =  Eigen::MatrixXd(rows, cols);

	for (size_t i = 0; i < rows * cols; i++) {
		reshapedMatrix << gradients[i];
	}

	return reshapedMatrix;
}


