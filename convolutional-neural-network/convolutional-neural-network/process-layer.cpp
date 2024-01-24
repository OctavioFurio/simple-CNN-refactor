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

    _convolutionOutput  =  _convolutionLayer.CalculateConvolution(input);
    return _convolutionOutput;
}



Eigen::MatrixXd ProcessLayer::CalculateConvolution(cv::Mat image)
{
    Eigen::MatrixXd input  =  Utils::ImageToMatrix( image );

    _convolutionOutput  =  _convolutionLayer.CalculateConvolution( input );
    return _convolutionOutput;
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
    _poolingOutput  =  _poolingLayer->FowardPooling( input );
    return _poolingOutput;
}


/*
std::vector<double> ProcessLayer::Flattening(Eigen::MatrixXd input)
{
    std::vector<double> vec;
    vec.reserve( input.size() );

    for (int i = 0; i < input.rows(); i++) {
        for (int j = 0; j < input.cols(); j++) {
            vec.push_back( input(i, j) );
        }
    }

    return vec;
}



Eigen::MatrixXd ProcessLayer::Reshape(std::vector<double> gradients, size_t rows, size_t cols)
{
    Eigen::MatrixXd reshapedMatrix  =  Eigen::MatrixXd(rows, cols);

    for (size_t i = 0; i < rows * cols; i++) {
        reshapedMatrix << gradients[i];
    }

    return reshapedMatrix;
}
*/



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





