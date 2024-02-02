#include "convolutional-neural-network.h"





CNN::CNN(
	int inputRows
	, int inputCols
	, std::vector<ProcessLayer> processLayers
	, std::vector<size_t> mlpArchitecture
	, IActivationFunction* activationFunction
	, double learningRate
) {

	_maxEphocs = 50000;
	_minError = -1.0;
	_errorEnergy = -1.0;

	_processLayers = processLayers;
	_learningRate = learningRate;

	int rows = inputRows;
	int cols = inputCols;

	for (auto processLayer : processLayers) {
		rows  =  (rows - processLayer.KernelRows() + 1)   /   processLayer.PoolingRows();
		cols  =  (cols - processLayer.KernelCols() + 1)   /   processLayer.PoolingCols();

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
	Eigen::MatrixXd processResult = Utils::BatchNormalization(input);

	for (auto& processLayer : _processLayers) {
		 Eigen::MatrixXd convolvedMatrix  =  processLayer.CalculateConvolution(processResult);
		 Eigen::MatrixXd activatedMatrix  =  processLayer.ApplayActivationFunction(convolvedMatrix);
		 Eigen::MatrixXd pooledMatrix	 =  processLayer.CalculatePooling( activatedMatrix );

		 processResult = pooledMatrix;
	}

	_reshapeRows = processResult.rows();
	_reshapeCols = processResult.cols();

	_flattedMatrix  =  Flattening( processResult );
	//std::cout << "\n_flattedMatrix\n" << processResult << "\n\n";
	_flattedMatrix  =  Utils::BatchNormalization(_flattedMatrix);
	_flattedMatrix.insert(_flattedMatrix.begin(), 1.0);
	
	std::vector<double> mlpOutput  =  _mlp.Forward( _flattedMatrix );


	return mlpOutput;
}



std::vector<double> CNN::Backward(std::vector<double> correctOutputs, Eigen::MatrixXd input)
{
	//std::vector<double> mlpInputs  =  Utils::BatchNormalization(_flattedMatrix);
	//mlpInputs.insert(mlpInputs.begin(), 1.0);

	std::vector<double> inputsGradient = _mlp.Backward(correctOutputs, _flattedMatrix);
	inputsGradient = std::vector<double>(inputsGradient.begin()+1, inputsGradient.end());

	Eigen::MatrixXd gradientFromNextLayer = Reshape(inputsGradient, _reshapeRows, _reshapeCols);



	// update hiden process layers
	for (size_t i = _processLayers.size() - 1; i > 0; i--) {
		Eigen::MatrixXd inputFromPreviousLayer  =  _processLayers[i-1].Output();
		Eigen::MatrixXd backwardPooling  =  _processLayers[i].PoolingBackward(gradientFromNextLayer);
		gradientFromNextLayer  =  _processLayers[i].ConvolutionBackward(inputFromPreviousLayer, backwardPooling, _learningRate);

	}

	// update fist layer
	input = Utils::BatchNormalization(input);
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
	//std::cout << "\n\n\n-------------------------------------------------------------------------\n";
	//for (auto& l : cnn._processLayers) {
	//
	//	os << "\n\nkernel: \n" << l.Kernel() << "\n\n";
	//	os << "\n\ngradient: \n" << l.Gradient() << "\n\n";
	//}
	//
	//std::vector<double> errors = cnn._mlp.Errors();
	//
	//double meanError = std::accumulate(errors.begin(), errors.end(), 0.0, [](double a, double b) { return a += std::abs(b); }) / errors.size();


	os << "\n\nERRORS:  " << cnn._error << "\n";

	return os;
}



void CNN::Training(std::vector<CnnTrainingData> trainigSet)
{
	size_t ephocs = 0;
	double errors = 1000000.0;
	double errorEnergy = 1000000.0;
	int trainingSetLenght = trainigSet.size();

	std::vector<double> _lastLayerErrors;

	while (ephocs <= _maxEphocs  &&  errors > _minError  && errorEnergy > _errorEnergy) {

		for (int i = 0; i < trainingSetLenght; i++) {
			Eigen::MatrixXd inputs  =  trainigSet[i].first;
			std::vector<double> labels  =  trainigSet[i].second;


			std::vector<double> lastLayaerOutput = Forward(inputs);
			_lastLayerErrors = Backward(labels, inputs);
		}

		//---------------------------
		// criterios de parada
		//---------------------------
		errors = 0.0;
		errorEnergy = 0.0;

		for (auto& e : _lastLayerErrors) { errors += std::abs(e);  errorEnergy += e * e; }

		errors = errors / _mlp._layers[_mlp._layersSize-1].NumberOfNeurons();
		errorEnergy = errorEnergy / 2.0;

		ephocs++;
	}
}



void CNN::Training(std::vector<CnnData> dataSet, std::function<std::vector<double>(int)> ParseLabelIndexToLastLayerOutput)
{
	int numberOfNeuronInLastLayer = _mlp._layers[_mlp._layers.size()-1].NumberOfNeurons();

	std::vector<CnnTrainingData> trainigSet;

	for (auto& t : dataSet) {
		std::vector<double> correctLastLayerOutputs = ParseLabelIndexToLastLayerOutput(t.labelIndex);

		assert(correctLastLayerOutputs.size() == numberOfNeuronInLastLayer);

		trainigSet.push_back({ t.inputs, correctLastLayerOutputs });
	}

	Training(trainigSet);
}



void CNN::Training(std::vector<CnnTrainingData> trainigSet, int callbackExecutionPeriod, std::function<void(void)> callback)
{
	size_t ephocs = 0;
	double errors = 100.0;
	double errorEnergy = 100.0;
	int trainingSetLenght = trainigSet.size();
	int minimalChangesCounter = 0;

	std::vector<double> _lastLayerErrors;


	while (ephocs <= _maxEphocs  &&  errors > _minError  &&  minimalChangesCounter != 20) {

		double iterationError = 0.0;

		for (int i = 0; i < trainingSetLenght; i++) {
			Eigen::MatrixXd inputs  =  trainigSet[i].first;
			std::vector<double> labels  =  trainigSet[i].second;
			

			std::vector<double> lastLayaerOutput = Forward(inputs);
			_lastLayerErrors = Backward(labels, inputs);
			

			iterationError  +=  ( std::accumulate( _lastLayerErrors.begin(), _lastLayerErrors.end(), 0.0, [](double acc, double val) { return acc + (val*val); }) / _lastLayerErrors.size());
		}

		//normalize convolution Kernel
		for (auto& processLayer : _processLayers) {
			processLayer.NormalizeKernel();
		}


		// shuffle trainig set
		std::random_device rd;
		std::mt19937 g(rd());
		std::shuffle(trainigSet.begin(), trainigSet.end(), g);



		//// ------------------
		//// coloque seu calback aqui
		//// ------------------

		if (ephocs % callbackExecutionPeriod == 0) { callback(); }

		//// ------------------
		

		_error = iterationError / trainingSetLenght;

		double currentErrors = _error;
		if (std::abs(errors - currentErrors) < 1.0e-10) { minimalChangesCounter++; } else { minimalChangesCounter = 0; }
		errors = currentErrors;
		

		ephocs++;

		//ephocs <= _maxEphocs  &&  errors > _minError  &&  minimalChangesCounter != 20
		if (!(ephocs <= _maxEphocs)) { std::cout << "\n\n[MAX EPHOC]\n\n"; }
		else if (!(errors > _minError)) { std::cout << "\n\n[MIN ERROR]\n\n"; }
		else if (!(minimalChangesCounter != 20)) { std::cout << "\n\n[ERROR NOT CHANGING]\n\n"; }

	}
}



void CNN::Training(
	std::vector<CnnData> dataSet
	, int callbackExecutionPeriod
	, std::function<std::vector<double>(int)> ParseLabelIndexToLastLayerOutput
	, std::function<void(void)> callback
){
	int numberOfNeuronInLastLayer = _mlp._layers[_mlp._layers.size()-1].NumberOfNeurons();

	std::vector<CnnTrainingData> trainigSet;
	for (auto& t : dataSet) {
		std::vector<double> correctLastLayerOutputs = ParseLabelIndexToLastLayerOutput(t.labelIndex);

		assert(correctLastLayerOutputs.size() == numberOfNeuronInLastLayer);

		trainigSet.push_back({ t.inputs, correctLastLayerOutputs });
	}

	Training(trainigSet, callbackExecutionPeriod, callback);
}



void CNN::MaxAcceptableError(double error)
{
	_minError = error;
}



std::vector<double> CNN::ProcessInput(Eigen::MatrixXd input)
{
	std::vector<double> givenOutputFromLastLayer = Forward(input);
	return givenOutputFromLastLayer;
}



std::vector<ProcessLayer> CNN::ProcessLayers()
{
	return _processLayers;
}

const double CNN::Error()
{
	return _error;
}



void CNN::LearningRate(double rate)
{
	_learningRate = rate;
}
