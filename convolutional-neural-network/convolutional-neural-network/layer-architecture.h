#pragma once


#include "neuron.h"



struct LayerArchitecture {

	size_t _qntNeurons;
	IActivationFunction* _activationFunction;
	double _learningRate;

	LayerArchitecture(size_t qntNeurons, IActivationFunction* activationFunction = new Tanh(), double learningRate = 0.03)
		: _qntNeurons(qntNeurons), _activationFunction(activationFunction), _learningRate(learningRate)
	{ }
};

