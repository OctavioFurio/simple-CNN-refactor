#pragma once

#include "neuron.h"


class Layer {

	private:
		std::vector<Neuron> _neurons;
		std::vector<double> _outputs;
		size_t _inputSize;
		size_t _layerIndex;

		IActivationFunction* _activationFunction;
		double _neuronLerningRate;


	public:
		Layer(size_t neuronQuantity, size_t inputSize, IActivationFunction* actFun = new Sigmoid(), size_t index = 0, double neuronLerningRate = 0.03);
		~Layer();

		std::vector<double> CalculateLayerOutputs(std::vector<double> inputs);
		double AccumulatedPropagatedErrorByNeuronAtIndex(int neuronIndex);
		std::vector<double> AccumulatedPropagatedErrorByPreviousLayer(int neuronsInPreviousLayer);
		size_t OutputSize();
		void UpdateHiddenLayerNeurons(std::vector<double> accumulatedPropagatedErrors, std::vector<double> inputs);
		std::vector<double> UpdateLastLayerNeurons(std::vector<double> correctOutputs, std::vector<double> inputs);
		int NumberOfNeurons();
		
		std::vector<double> LayerOutputs();
		size_t LayerIndex();

		Neuron& operator[](int neuronIndex);
		friend std::ostream& operator<<(std::ostream& os, Layer layer);
		Json ToJson() const;
		Layer LoadWeightsFromJson(const Json& j);

		std::vector<double> EvaluateInputs(std::vector<double> inputs);   // calcula o output dessa camada sem salvar os valores
		std::vector<Neuron> Neurons() { return _neurons; }

};

