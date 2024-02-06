#include "layer.h"



Layer::Layer(size_t neuronQuantity, size_t inputSize, IActivationFunction* actFun, size_t index, double neuronLerningRate)
	: _inputSize(inputSize), _layerIndex(index), _activationFunction(actFun), _neuronLerningRate(neuronLerningRate)
{
	_neurons = std::vector<Neuron>(neuronQuantity, Neuron(inputSize, actFun, neuronLerningRate));

	size_t layerOutputSize = _neurons.size() + 1;
	_outputs =  std::vector<double>(layerOutputSize, 1.0);
}



Layer::~Layer()
{
}



std::vector<double> Layer::CalculateLayerOutputs(std::vector<double> inputs)
{
	int outputIndex = 1;

	inputs = Utils::BatchNormalization( std::vector<double>(inputs.begin()+1, inputs.end()) );
	inputs.insert(inputs.begin(), 1.0);

	for (auto& neuron : _neurons) {
		_outputs[outputIndex++]  =  neuron.CalculateOutput(inputs);
	}

	return _outputs;
}



double Layer::AccumulatedPropagatedErrorByNeuronAtIndex(int neuronIndex)
{
	double sum = 0.0;

	for (auto& neuron : _neurons) {
		double gradient = neuron.Gradient();
		double weight = neuron[neuronIndex];

		sum  +=  gradient * weight;
	}

	return sum;
}



std::vector<double> Layer::AccumulatedPropagatedErrorByPreviousLayer(int neuronsInPreviousLayer)
{
	std::vector<double> accumulatedPropagatedErrors =  std::vector<double>((size_t)neuronsInPreviousLayer, 0.0);

	for (int i = 0; i < neuronsInPreviousLayer; i++) {
		accumulatedPropagatedErrors[i]  =  AccumulatedPropagatedErrorByNeuronAtIndex(i);
	}

	return accumulatedPropagatedErrors;
}



size_t Layer::OutputSize()
{
	return _neurons.size() + 1;
}



void Layer::UpdateHiddenLayerNeurons(std::vector<double> accumulatedPropagatedErrors, std::vector<double> inputs)
{
	assert(_neurons.size() == accumulatedPropagatedErrors.size());

	int errorIndex = 0;

	for (auto& neuron : _neurons) {
		neuron.CalculateGradient( accumulatedPropagatedErrors[errorIndex++] );
		neuron.UpdateWeights(inputs);
	}
}



std::vector<double> Layer::UpdateLastLayerNeurons(std::vector<double> correctOutputs, std::vector<double> inputs)
{
	assert(correctOutputs.size() == _neurons.size());

	int correctOutputsIndex = 0;
	size_t numberOfNeurons = _neurons.size();

	std::vector<double> errors  =  std::vector<double>(numberOfNeurons, 1.0);

	for (auto& neuron : _neurons) {
		double valueThatShouldBe  =  correctOutputs[correctOutputsIndex];

		errors[correctOutputsIndex++] = neuron.CalculateError(valueThatShouldBe);
		neuron.UpdateWeights(inputs);
	}

	return errors;
}



Neuron& Layer::operator[](int neuronIndex)
{
	return (Neuron&) _neurons[neuronIndex];
}



Json Layer::ToJson() const
{
	Json layerJson;
	for (const auto& neuron : _neurons) {
		layerJson["neurons"].push_back(neuron.ToJson());
	}

	return { {"layer", layerJson} };
}



Layer Layer::LoadWeightsFromJson(const Json& j)
{
	int neuronIndex = 0;
	for (const auto& neuronJson : j.at("layer").at("neurons")) {
		Neuron n = Neuron(_inputSize, _activationFunction, _neuronLerningRate);
		n.LoadWeightsFromJson(neuronJson);

		int weightSize = n.Weight().size();
		
		(*this)._neurons[neuronIndex] = n;

		neuronIndex++;
	}

	return (*this);
}



std::vector<double> Layer::EvaluateInputs(std::vector<double> inputs)
{
	std::vector<double> thisLayerOutput = std::vector<double>( (size_t)NumberOfNeurons()+1, 1.0 );

	int outputIndex = 1;

	for (auto& neuron : _neurons) {
		thisLayerOutput[outputIndex++]  =  neuron.EvaluateInputs(inputs);
	}

	return thisLayerOutput;
}



int Layer::NumberOfNeurons()
{
	return _neurons.size();
}




std::vector<double> Layer::LayerOutputs()
{
	return _outputs;
}

size_t Layer::LayerIndex()
{
	return _layerIndex;
}



std::ostream& operator<<(std::ostream& os, Layer layer)
{
	for (auto& n : layer._neurons) {
		os << n.Gradient() << "  ";
	}

	//os << layer._neurons[0] << "  ";

	return os;
}
