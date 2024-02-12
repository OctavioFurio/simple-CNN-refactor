#pragma once

#include "activation-function.h"
#include "layer.h"
#include "layer-architecture.h"
#include "mlp-input-data.h"
#include <fstream>
#include <functional>
#include <string>
#include <utility>

using TrainigData = std::pair<std::vector<double>, std::vector<double>>;

class MLP {

private:
	std::vector<Layer> _layers;
	std::vector<size_t> _mlpArchitecture;
	IActivationFunction* _activationFunction;

	size_t _inputSize;
	double _learningRate;
	int _layersSize;

	size_t _maxEphocs;
	double _minError;
	double _errorEnergy;

	std::vector<double> _lastLayerErrors;

	MLP();

	void LoadFromJson(const Json& j);
	void LoadWeightsFromJson(const char* filePath);
	IActivationFunction* StringToActivationFunction(std::string functionName);

	friend class CNN;


public:
	MLP(size_t inputSize, std::vector<size_t> mlpArchitecture, IActivationFunction* _activationFunction = new Tanh(), double learningRate = 0.03);
	MLP(size_t inputSize, std::vector<LayerArchitecture> layersArchitecture);
	MLP(std::string filePath);
	~MLP();

	void Training(std::vector<TrainigData> trainigSet);
	void Training(std::vector<TrainigData> trainigSet, int callbackExecutionPeriod, std::function<void(void)> callback);    // callbak a ser executado apos cada iteraçao de aprendizado

	void Training(std::vector<DATA> dataSet, std::function<std::vector<double>(int)> ParseLabelIndexToLastLayerOutput);

	void Training(
		std::vector<DATA> trainigSet
		, int callbackExecutionPeriod
		, std::function<std::vector<double>(int)> ParseLabelIndexToLastLayerOutput
		, std::function<void(void)> callback
	);

	void Processing(
		std::vector< std::vector<double> > processingSet
		, std::function<int(std::vector<double>)> ParseOutputToLabelIndex
		, std::function<void(int)> AfterProcessing
	);

	void AcceptableErrorEnergy(double error);
	void AcceptableMeanError(double error);
	void MaximunEphocs(size_t ephocs);

	std::vector<double> Forward(std::vector<double> inputs);
	std::vector<double> Backward(std::vector<double> correctOutputs, std::vector<double> inputs);            

	Layer& operator[](int index);
	friend std::ostream& operator<<(std::ostream& os, MLP mlp);

	Json ToJson() const;
	void GenerateJsonFile(const char* filePath);

	std::vector<double> EvaluateInputs(std::vector<double> inputs);
	std::vector<double> Errors();
};