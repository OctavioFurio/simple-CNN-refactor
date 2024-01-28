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
		rows  =  std::ceil( (rows - processLayer.KernelRows() + 1) / processLayer.PoolingRows() );
		cols  =  std::ceil( (cols - processLayer.KernelCols() + 1) / processLayer.PoolingCols() );

		assert( rows > 0   &&   cols > 0 );
	}


	size_t mlpInputSize  =  rows * cols;

	_mlp  =  MLP(mlpInputSize, mlpArchitecture, activationFunction, learningRate);
}



CNN::~CNN()
{
}



std::vector<double> CNN::Forward(Eigen::MatrixXd input)
{
	Eigen::MatrixXd processResult = input;

	for (auto& processLayer : _processLayers) {
		 Eigen::MatrixXd convolvedMatrix  =  processLayer.CalculateConvolution(processResult);
		 Eigen::MatrixXd activatedMatrix  =  processLayer.ApplayActivationFunction(convolvedMatrix);
		 Eigen::MatrixXd pooledMatrix	 =  processLayer.CalculatePooling( activatedMatrix );

		 
		 //std::vector<double> flattedPooledMatrix = Flattening(pooledMatrix);
		 //std::vector<double> normalizedMatrix = Utils::BatchNormalization(flattedPooledMatrix);
		 //processResult  =  Reshape(normalizedMatrix, pooledMatrix.rows(), pooledMatrix.cols());

		 processResult = pooledMatrix;
	}

	_reshapeRows = processResult.rows();
	_reshapeCols = processResult.cols();

	_flattedMatrix  =  Flattening( processResult );
	std::vector<double> normalizedVector = Utils::BatchNormalization(_flattedMatrix);//Utils::NormalizeVector(_flattedMatrix, -1.0, 1.0);


	normalizedVector.insert(normalizedVector.begin(), 1.0);
	
	std::vector<double> mlpOutput  =  _mlp.Forward( normalizedVector );


	return mlpOutput;
}



std::vector<double> CNN::Backward(std::vector<double> correctOutputs, Eigen::MatrixXd input)
{
	std::vector<double> mlpInputs  =  Utils::BatchNormalization(_flattedMatrix);
	mlpInputs.insert(mlpInputs.begin(), 1.0);

	std::vector<double> inputsGradient = _mlp.Backward(correctOutputs, mlpInputs);
	inputsGradient = std::vector<double>(inputsGradient.begin()+1, inputsGradient.end());

	Eigen::MatrixXd gradientFromNextLayer = Reshape(inputsGradient, _reshapeRows, _reshapeCols);



	// update hiden process layers
	for (size_t i = _processLayers.size() - 1; i > 0; i--) {
		Eigen::MatrixXd inputFromPreviousLayer  =  _processLayers[i-1].Output();

		std::vector<double> flattedInputFromPreviousLayer = Flattening(inputFromPreviousLayer);
		std::vector<double> normalizedMatrix = Utils::BatchNormalization(flattedInputFromPreviousLayer);
		inputFromPreviousLayer  =  Reshape(normalizedMatrix, inputFromPreviousLayer.rows(), inputFromPreviousLayer.cols());

		Eigen::MatrixXd backwardPooling  =  _processLayers[i].PoolingBackward(gradientFromNextLayer);
		gradientFromNextLayer  =  _processLayers[i].ConvolutionBackward(inputFromPreviousLayer, backwardPooling, _learningRate);

	}

	// update fist layer
	Eigen::MatrixXd backwardPooling  =  _processLayers[0].PoolingBackward(gradientFromNextLayer);
	gradientFromNextLayer  =  _processLayers[0].ConvolutionBackward(input, backwardPooling, _learningRate);



	std::vector<double> mlpErrors = _mlp.Errors();
	return mlpErrors;
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
	Eigen::MatrixXd reshapedMatrix  =  Eigen::MatrixXd::Zero(rows, cols);

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			reshapedMatrix(i,j)  =  gradients[i*cols + j];
		}
	}

	//std::cout << reshapedMatrix << "\n\n";

	return reshapedMatrix;
}



std::ostream& operator<<(std::ostream& os, CNN cnn)
{
	for (auto l : cnn._processLayers) {
		os << "\n\n" << l.Kernel() << "\n";
	}

	std::vector<double> errors = cnn._mlp.Errors();

	double meanError = std::accumulate(errors.begin(), errors.end(), 0.0, [](double a, double b) { return std::abs(b); }) / errors.size();

	os << "\n\nERRORS:  " << meanError << "\n";

	return os;
}

