#include "neuron.h"


Neuron::Neuron(size_t inputSize, IActivationFunction* actFun, double leraningRate)
    : _inputSize(inputSize+1), _learningRate(leraningRate), activationFunction(actFun)
{
    _error = 0.0;
    _output = 0.0;
    _u = 0.0;
    _gradient = 0.0;

    _weights = std::vector<double>(_inputSize, 1.0);

    /*for (auto& weight : _weights) {
        weight  =  RandomNormalDistributionvalue(-1.0, 1.0);
    }*/
}



Neuron::~Neuron()
{
}



double Neuron::RandomNormalDistributionvalue(double min, double max)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> distribuicao(min, max);
    return distribuicao(gen);
}



double Neuron::ScalarProduct(std::vector<double> inputs, std::vector<double> weights)
{
    assert(inputs.size() == weights.size());

    double sum = 0.0;

    for (size_t i = 0; i < weights.size(); i++) {
        sum += inputs[i] * weights[i];
    }

    return sum;
}



double Neuron::CalculateOutput(std::vector<double> inputs)
{
    _u = ScalarProduct(inputs, _weights);
    _output = activationFunction->f(_u);
    return _output;
}



// weight[j] = weight[j] + learningRate * ( d - y ) * input[j];
double Neuron::CalculateError(double correctValue)
{
    double error = correctValue - _output;
    double du = activationFunction->df(_u);

    _gradient = error * du;

    return error;
}



// erro propagado acumulado do neuronio 2
// accumulatedPropagatedError = somatorio neuronio i = 0->n { gradiente_i * peso_i_2 }
double Neuron::CalculateGradient(double accumulatedPropagatedError)
{
    double du = activationFunction->df(_u);

    _gradient  =  accumulatedPropagatedError * du;

    return _gradient;
}



void Neuron::UpdateWeights(std::vector<double> receivedInputs)
{
    assert(receivedInputs.size() == _weights.size());

    for (size_t i = 0; i < _weights.size(); i++) {
        _weights[i]  +=  _learningRate * _gradient * receivedInputs[i];
    }
}



double& Neuron::operator[](int weightIndex)
{
    return (double&) _weights[weightIndex];
}



Neuron Neuron::operator=(const Neuron& neuron)
{
  
    if (this != &neuron) {
        _weights = neuron._weights;
        activationFunction = neuron.activationFunction;
        _inputSize = neuron._inputSize;
        _error = neuron._error;
        _output = neuron._output;
        _u = neuron._u;
        _gradient = neuron._gradient;
        _learningRate = neuron._learningRate;
    }
    return *this;
}



Json Neuron::ToJson() const
{
    Json json = {
        {"bias", _weights[0] },
        {"weights", _weights}
    };

    return json;
}



std::vector<double> Neuron::LoadWeightsFromJson(const Json& j)
{
    (*this)._weights = j.at("weights").get<std::vector<double>>();
    return (*this)._weights;
}



double Neuron::EvaluateInputs(std::vector<double> inputs)
{
    double u = ScalarProduct(inputs, _weights);
    double output = activationFunction->f(_u);
    return output;
}



const double Neuron::Bias()
{
    return _weights[0];
}

const double Neuron::Error()
{
    return _error;
}

const double Neuron::Output()
{
    return _output;
}

const double Neuron::U()
{
    return _u;
}

const double Neuron::Gradient()
{
    return _gradient;
}

const std::vector<double> Neuron::Weight()
{
    return _weights;
}



std::ostream& operator<<(std::ostream& os, Neuron neuron)
{
    os << "[ ";
    for (auto& w : neuron._weights) {
        os << w << " ; ";
    }
    os << "\b\b ]";

    return os;
}
