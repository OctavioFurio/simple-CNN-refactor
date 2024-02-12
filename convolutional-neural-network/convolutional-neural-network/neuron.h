#pragma once

#include "activation-function.h"
#include <cassert>
#include <iostream>
#include <nlohmann/json.hpp>
#include <random>
#include <string>
#include <vector>

using Json = nlohmann::json;

class Neuron {

private:
	std::vector<double> _weights;
	IActivationFunction* activationFunction;

	size_t _inputSize;
	double _error;
	double _output;
	double _u;
	double _gradient;
	double _learningRate;

	double RandomNormalDistributionvalue(double min, double max);
	double ScalarProduct(std::vector<double> inputs, std::vector<double> weights);

public:
	Neuron(size_t inputSize, IActivationFunction* actFun = new Tanh(), double leraningRate = 0.03);
	~Neuron();

	double CalculateOutput(std::vector<double> inputs);
	double CalculateError(double correctValue);
	double CalculateGradient(double accumulatedPropagatedError);
	void UpdateWeights(std::vector<double> receivedInputs);

	const double Bias();
	const double Error();
	const double Output();
	const double U();
	const double Gradient();
	const std::vector<double> Weight();

	double& operator[](int weightIndex);
	friend std::ostream& operator<<(std::ostream& os, Neuron neuron);
	Neuron operator=(const Neuron& neuron);

	Json ToJson() const;
	std::vector<double> LoadWeightsFromJson(const Json& j);
	double EvaluateInputs(std::vector<double> inputs);
};