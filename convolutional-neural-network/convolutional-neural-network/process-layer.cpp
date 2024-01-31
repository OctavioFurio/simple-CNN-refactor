#include "process-layer.h"




ProcessLayer::ProcessLayer(Eigen::MatrixXd& kernel, int padding, IActivationFunction* actFunc, IPooling* pooling)
    : _activationFunction(actFunc), _poolingLayer(pooling)
{
    
    _convolutionLayer  =  ConvolutionLayer(kernel, padding);
    
}



ProcessLayer::ProcessLayer(std::vector<std::vector<double>>& kernel, int padding, IActivationFunction* actFunc, IPooling* pooling)
    : _activationFunction(actFunc), _poolingLayer(pooling)
{
    _convolutionLayer  =  ConvolutionLayer(kernel, padding);

   
}



ProcessLayer::~ProcessLayer()
{
}



Eigen::MatrixXd ProcessLayer::CalculateConvolution(Eigen::MatrixXd input)
{
    std::vector<double> flattedIncomeInput = Utils::FlatMatrix(input);
    std::vector<double> normalizedIncomeInput = Utils::BatchNormalization(flattedIncomeInput);
    input  =  Utils::ReshapeMatrix(normalizedIncomeInput, input.rows(), input.cols());

    // por que nao armazena
    Eigen::MatrixXd concolvedMatrix  =  _convolutionLayer.CalculateConvolution(input);

    _convolutionOutput  =  Eigen::MatrixXd(concolvedMatrix);

    return _convolutionOutput;
}



Eigen::MatrixXd ProcessLayer::CalculateConvolution(cv::Mat image)
{
    Eigen::MatrixXd input  =  Utils::ImageToMatrix( image );

    Eigen::MatrixXd& concolvedMatrix  =  _convolutionLayer.CalculateConvolution( input );
    _convolutionOutput  =  concolvedMatrix;

    return _convolutionOutput;
}



Eigen::MatrixXd ProcessLayer::ConvolutionBackward(Eigen::MatrixXd& input, Eigen::MatrixXd& incomeGradient, double learningRate)
{
    Eigen::MatrixXd gradientOfLostWithRespctToInput = _convolutionLayer.Backward(input, incomeGradient, learningRate);
    return gradientOfLostWithRespctToInput;
}



Eigen::MatrixXd ProcessLayer::ApplayActivationFunction(Eigen::MatrixXd input)
{
    for (int i = 0; i < input.rows(); i++) {
        for (int j = 0; j < input.cols(); j++) {
            input(i,j)  =  _activationFunction->f( input(i,j) );
        }
    }

    return input;
}



Eigen::MatrixXd ProcessLayer::CalculatePooling(Eigen::MatrixXd input)
{
    Eigen::MatrixXd& pooledMatrix  =  _poolingLayer->FowardPooling( input );

    _poolingOutput  =  pooledMatrix;

    return _poolingOutput;
}



Eigen::MatrixXd ProcessLayer::PoolingBackward(Eigen::MatrixXd input)
{
    Eigen::MatrixXd backwardPooling = _poolingLayer->BackwardPooling(input);
    return backwardPooling;
}



Eigen::MatrixXd ProcessLayer::ConvolutionOutout()
{
    return _convolutionOutput;
}



Eigen::MatrixXd ProcessLayer::PoolingOutput()
{
    return _poolingOutput;
}



Eigen::MatrixXd ProcessLayer::Output()
{
    return _poolingOutput;
}



int ProcessLayer::KernelRows()
{
    return _convolutionLayer.Kernel().rows();
}

int ProcessLayer::KernelCols()
{
    return _convolutionLayer.Kernel().cols();
}



int ProcessLayer::PoolingRows()
{
    return _poolingLayer->Rows();
}

int ProcessLayer::PoolingCols()
{
    return _poolingLayer->Cols();
}



Eigen::MatrixXd ProcessLayer::Kernel()
{
    return _convolutionLayer.Kernel();
}



Eigen::MatrixXd ProcessLayer::Gradient()
{
    return _convolutionLayer.Gradient();
}





