#include <opencv2/opencv.hpp>
#include "multy-layer-perceptron.h"




MLP::MLP()
{

}



MLP::MLP(size_t inputSize, std::vector<size_t> mlpArchitecture, IActivationFunction* activationFunction, double learningRate)
{
	_maxEphocs = 50000000;
	_minError = -1.0;
	_errorEnergy = -1.0;
	_layersSize  =  mlpArchitecture.size();
	_mlpArchitecture = mlpArchitecture;
	_learningRate = learningRate;
	_inputSize = inputSize;
	_activationFunction = activationFunction;
	size_t currentInputSize  =  inputSize;
	size_t layerIndex = 0;

	for (auto neuronsInLayer : mlpArchitecture) {
		_layers.push_back(  Layer(neuronsInLayer, currentInputSize, activationFunction, layerIndex++, learningRate)  );
		currentInputSize = neuronsInLayer;
	}
}



MLP::MLP(size_t inputSize, std::vector<LayerArchitecture> layersArchitecture)
{
	_maxEphocs = 50000000;
	_minError = -1.0;
	_errorEnergy = -1.0;
	_layersSize  =  layersArchitecture.size();
	//_mlpArchitecture = ;
	_learningRate = 0.03;
	_inputSize = inputSize;
	_activationFunction = new Tanh();
	size_t currentInputSize  =  inputSize;
	size_t layerIndex = 0;

	for (auto layerArchitecture : layersArchitecture) {
		size_t neuronsInLayer = layerArchitecture._qntNeurons;
		IActivationFunction* activationFunction = layerArchitecture._activationFunction;
		double learningRate = layerArchitecture._learningRate;

		_layers.push_back( Layer(neuronsInLayer, currentInputSize, activationFunction, layerIndex++, learningRate) );
		currentInputSize = neuronsInLayer;
	}
}



MLP::MLP(std::string filePath)
{
	const char* filePathStr = filePath.c_str();
	std::ifstream jsonFile(filePathStr);

	Json json;
	jsonFile >> json;
	jsonFile.close();

	LoadFromJson(json);
}



MLP::~MLP()
{
}



std::vector<double> MLP::Forward(std::vector<double> inputs)
{
	std::vector<double> layerInput = inputs;

	for (auto& layer : _layers) {
		std::vector<double> nextLayerInput = layer.CalculateLayerOutputs(layerInput);
		layerInput = nextLayerInput;
	}

	std::vector<double> lastLayerOutput  =  _layers[_layersSize-1].LayerOutputs();

	return std::vector<double>( lastLayerOutput.begin()+1, lastLayerOutput.end() );
}



std::vector<double> MLP::Backward(std::vector<double> correctOutputs, std::vector<double> inputs)
{
	int layerIndex = _layers.size() - 1;   // comecando de tras para frente 
	

	// atualizando a ultima camada
	std::vector<double> InputForLastLayer  =  (layerIndex-1 < 0) ? inputs : _layers[layerIndex-1].LayerOutputs();
	_lastLayerErrors = _layers[layerIndex--].UpdateLastLayerNeurons(correctOutputs, InputForLastLayer);


	// atualiza as outras camadas
	for (layerIndex; layerIndex > 0; layerIndex--) {

		int neuronsInCurrentLayer  =  _layers[layerIndex].NumberOfNeurons();
		std::vector<double> accumulatedPropagatedErrors  =  _layers[layerIndex+1].AccumulatedPropagatedErrorByPreviousLayer(neuronsInCurrentLayer);
		std::vector<double> inputForCurrentLayer  =  _layers[layerIndex-1].LayerOutputs();

		_layers[layerIndex].UpdateHiddenLayerNeurons(accumulatedPropagatedErrors, inputForCurrentLayer);
	}


	// atualiza a primeira camada
	int neuronsInCurrentLayer  =  _layers[0].NumberOfNeurons();
	std::vector<double> accumulatedPropagatedErrors  =  _layers[layerIndex+1].AccumulatedPropagatedErrorByPreviousLayer(neuronsInCurrentLayer);
	_layers[0].UpdateHiddenLayerNeurons(accumulatedPropagatedErrors, inputs);

	std::vector<double> fistLayerInputGradient  =  _layers[0].AccumulatedPropagatedErrorByPreviousLayer(inputs.size());

	return fistLayerInputGradient;
}



void MLP::Training(std::vector<TrainigData> trainigSet)
{
	size_t ephocs = 0;
	double errors = 1000000.0;
	double errorEnergy = 1000000.0;
	int trainingSetLenght = trainigSet.size();

	while (ephocs <= _maxEphocs  &&  errors > _minError  && errorEnergy > _errorEnergy) {

		if (ephocs % 1000 == 0) { std::cout << *this; }

		for (int i = 0; i < trainingSetLenght; i++) {
			std::vector<double> inputs  =  trainigSet[i].first;
			std::vector<double> labels  =  trainigSet[i].second;

			inputs.insert(inputs.begin(), 1.0);

			std::vector<double> lastLayaerOutput = Forward(inputs);
			Backward(labels, inputs);
		}

		//---------------------------
		// criterios de parada
		//---------------------------
		
		// mean error 
		errors = 0.0;
		errorEnergy = 0.0;
		for (auto& e : _lastLayerErrors) { errors += std::abs(e);  errorEnergy += e * e; }
		errors = errors / _layers[_layersSize-1].NumberOfNeurons();
		errorEnergy = errorEnergy / 2.0;

		// ephocs
		ephocs++;
	}

}



void MLP::Training(std::vector<TrainigData> trainigSet, int callbackExecutionPeriod, std::function<void(void)> callback)
{
	size_t ephocs = 0;
	double errors = 1000000.0;
	double errorEnergy = 1000000.0;
	int trainingSetLenght = trainigSet.size();
	int minimalChangesCounter = 0;

	while (ephocs <= _maxEphocs  &&  errors > _minError  && errorEnergy > _errorEnergy  &&  minimalChangesCounter != 20000) {

		for (int i = 0; i < trainingSetLenght; i++) {
			std::vector<double> inputs  =  trainigSet[i].first;
			std::vector<double> labels  =  trainigSet[i].second;

			inputs.insert(inputs.begin(), 1.0);                                     // <-- 1.0 que sera multiplicado pelo bias no produto vetorial

			std::vector<double> lastLayaerOutput = Forward(inputs);
			Backward(labels, inputs);
		}


		/// ------------------
		/// imagem/video do mlp aprendendo - coloque seu calback aqui
		/// ------------------

		if (ephocs % callbackExecutionPeriod == 0) { callback(); }

		/// ------------------

		double currentErrors = 0.0;
		errorEnergy = 0.0;

		for (auto& e : _lastLayerErrors) { currentErrors += std::abs(e);  errorEnergy += e * e; }

		currentErrors = currentErrors / _layers[_layersSize-1].NumberOfNeurons();
		errorEnergy = errorEnergy / 2.0;

		if (std::abs(errors - currentErrors) < 1.0e-10) { minimalChangesCounter++; }
		else { minimalChangesCounter = 0; }

		errors = currentErrors; 

		ephocs++;
	}

}



void MLP::Training(std::vector<DATA> dataSet, std::function<std::vector<double>(int)> ParseLabelIndexToLastLayerOutput)
{
	int numberOfNeuronInLastLayer = _layers[_layers.size()-1].NumberOfNeurons();

	std::vector<TrainigData> trainigSet;
	for (auto& t : dataSet) {
		std::vector<double> correctLastLayerOutputs = ParseLabelIndexToLastLayerOutput(t.labelIndex);
		
		assert(correctLastLayerOutputs.size() == numberOfNeuronInLastLayer);

		trainigSet.push_back( { t.inputs, correctLastLayerOutputs } );
	}

	Training(trainigSet);
}



void MLP::Training(
	std::vector<DATA> dataSet
	, int callbackExecutionPeriod
	, std::function<std::vector<double>(int)> ParseLabelIndexToLastLayerOutput
	, std::function<void(void)> callback
) {
	int numberOfNeuronInLastLayer = _layers[_layers.size()-1].NumberOfNeurons();

	std::vector<TrainigData> trainigSet;
	for (auto& t : dataSet) {
		std::vector<double> correctLastLayerOutputs = ParseLabelIndexToLastLayerOutput(t.labelIndex);

		assert(correctLastLayerOutputs.size() == numberOfNeuronInLastLayer);

		trainigSet.push_back({ t.inputs, correctLastLayerOutputs });
	}

	Training(trainigSet, callbackExecutionPeriod, callback);
}



void MLP::Processing(
	std::vector<std::vector<double>> processingSet
	, std::function<int(std::vector<double>)> ParseOutputToLabelIndex
	, std::function<void(int)> AfterProcessing
) {

	std::cout << *this << "\n\n";

	for (auto& inputs : processingSet) {

		inputs.insert(inputs.begin(), 1.0);

		std::vector<double> output = Forward(inputs);
		int labelIndex = ParseOutputToLabelIndex( output );
		AfterProcessing( labelIndex );
	}
}



Layer& MLP::operator[](int index)
{
	return _layers[index];
}



Json MLP::ToJson() const
{
	Json mlpJson;

	mlpJson["learningRate"] = _learningRate;
	mlpJson["inputSize"] = _inputSize;
	mlpJson["architecture"] = _mlpArchitecture;
	mlpJson["activation-funtion"] = _activationFunction->ToString();

	for (auto& layer : _layers) {
		mlpJson["layers"].push_back(layer.ToJson());
	}

	return mlpJson;
}



void MLP::GenerateJsonFile(const char* filePath)
{
	Json json = ToJson();
	
	std::ofstream arquivoSaida(filePath);


	if (arquivoSaida.is_open()) {
		arquivoSaida << json.dump(4);
		arquivoSaida.close();
	} else {
		std::cerr << "\n\n[ERROR]: could not open file !!! \n\n";
	}
}



std::vector<double> MLP::EvaluateInputs(std::vector<double> inputs)
{
	std::vector<double> layerInput = inputs;

	for (auto& layer : _layers) {

		int layerNeuronIndex = 1;
		std::vector<double> nextLayerInput = std::vector<double>((size_t)layer.NumberOfNeurons()+1, 1.0);

		for (auto& neuron : layer.Neurons()) {
			nextLayerInput[layerNeuronIndex++] = neuron.EvaluateInputs(layerInput);
		}
			 
		layerInput = nextLayerInput;
	}

	return std::vector<double>(layerInput.begin()+1, layerInput.end());
}



std::vector<double> MLP::Errors()
{
	return _lastLayerErrors;
}



void MLP::LoadFromJson(const Json& j)
{
	std::vector<size_t> architecture = j["architecture"].get<std::vector<size_t>>();
	size_t inputSize = j["inputSize"].get<size_t>();
	double learningRate = j["learningRate"].get<double>();

	std::string funcName = j["activation-funtion"].get<std::string>();
	IActivationFunction* actFunc = StringToActivationFunction(funcName);

	// MLP::MLP(std::vector<size_t> mlpArchitecture, size_t inputSize, IActivationFunction* activationFunction, double learningRate)
	(*this) = MLP(inputSize, architecture, actFunc, learningRate);

	int layerIndex = 0;
	int neuronIndex = 0;

	for (const auto& layer : j.at("layers")) {
		for (const auto& neuron : layer.at("layer").at("neurons")) {
			_layers[layerIndex][neuronIndex++].LoadWeightsFromJson(neuron);
		}
		neuronIndex = 0;
		layerIndex++;
	}

	(*this);
}



void MLP::LoadWeightsFromJson(const char* filePath)
{
	std::ifstream jsonFile(filePath);

	Json json;
	jsonFile >> json;
	jsonFile.close();

	LoadFromJson( json );
}



IActivationFunction* MLP::StringToActivationFunction(std::string functionName)
{
	if (functionName == "ReLU") { return new ReLU(); }
	else if (functionName == "LeakyReLU") { return new LeakyReLU(); }
	else if (functionName == "Tanh") { return new Tanh(); }
	else if (functionName == "NormalizedTanh") { return new NormalizedTanh(); }
	else if (functionName == "Sigmoid") { return new Sigmoid(); }
	else if (functionName == "AdaptedSigmoid") { return new AdaptedSigmoid(); }
	else if (functionName == "Linear") { return new Linear(); }
	else { return nullptr; }
}



void MLP::AcceptableErrorEnergy(double error)
{
	_errorEnergy = error;
}



void MLP::AcceptableMeanError(double error)
{
	_minError = error;
}



void MLP::MaximunEphocs(size_t ephocs)
{
	_maxEphocs = ephocs;
}



std::ostream& operator<<(std::ostream& os, MLP mlp)
{
	int layerIndex = 0;
	
	for (auto& layer : mlp._layers) {
		std::cout << "LAYER: " << layerIndex << "\n";
		std::cout << layer << "\n\n";
		layerIndex++;
	}

	std::cout << "\n\nERRORS:\n";
	double meanError = 0.0;
	for (auto& error : mlp._lastLayerErrors) { meanError += std::abs(error); }
	std::cout << meanError / (double)mlp._lastLayerErrors.size();

	std::cout << "\n----------------------------------------------\n\n\n";

	return os;
}
