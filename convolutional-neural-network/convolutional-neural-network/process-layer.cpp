#include "process-layer.h"




ProcessLayer::ProcessLayer(Eigen::MatrixXd& kernel, int padding, IActivationFunction* actFunc, IPooling* pooling)
    : _kernel(kernel), _activationFunction(actFunc), _poolingOperation(pooling)
{
    _bias = 1.0;
}



ProcessLayer::~ProcessLayer()
{
}



Eigen::MatrixXd ProcessLayer::CalculateConvolution(Eigen::MatrixXd input)
{
    Eigen::MatrixXd convolutedInput  =  Convolution2D(input, _kernel, _padding);
    return convolutedInput;
}



Eigen::MatrixXd ProcessLayer::CalculateConvolution(cv::Mat image)
{
    int i = 0;
    Eigen::MatrixXd input  =  ImageToMatrix(image);
    Eigen::MatrixXd convolutedInput  =  Convolution2D(input, _kernel, _padding);
    //std::cout << "\n\n" << input << "\n\n";
    return convolutedInput;
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
    Eigen::MatrixXd pooledMatrix  =  _poolingOperation->Pooling( input );
    return pooledMatrix;
}



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



Eigen::MatrixXd ProcessLayer::Convolution2D(Eigen::MatrixXd input, Eigen::MatrixXd kernel)
{
    const int kernelRows = kernel.rows();
    const int kernelCols = kernel.cols();
    const int rows = (input.rows() - kernelRows) + 1;
    const int cols = (input.cols() - kernelCols) + 1;

    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(rows, cols);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double sum = input.block(i, j, kernelRows, kernelCols).cwiseProduct(kernel).sum();
            result(i, j) = sum;
        }
    }

    return result;
}



Eigen::MatrixXd ProcessLayer::Convolution2D(Eigen::MatrixXd input, Eigen::MatrixXd kernel, int padding)
{
    int kernelRows = kernel.rows();
    int kernelCols = kernel.cols();
    int rows = input.rows() - kernelRows + 2*padding + 1;
    int cols = input.cols() - kernelCols + 2*padding + 1;

    Matrix padded = Matrix::Zero(input.rows() + 2*padding, input.cols() + 2*padding);
    padded.block(padding, padding, input.rows(), input.cols()) = input;
    

    Matrix result = Matrix::Zero(rows, cols);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double sum = padded.block(i, j, kernelRows, kernelCols).cwiseProduct(kernel).sum();
            result(i, j) = sum;
        }
    }

    return result;
}



Eigen::MatrixXd ProcessLayer::ImageToMatrix(cv::Mat mat)
{
    Eigen::MatrixXd matrix = Eigen::MatrixXd(mat.rows, mat.cols);

    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols; j++) {
            uchar pixel  =  mat.at<uchar>(i, j);
            matrix(i, j) =  (double)pixel;
        }
    }

    return matrix;
}



cv::Mat ProcessLayer::MatrixToImage(Eigen::MatrixXd matrix)
{
    cv::Mat mat  =  cv::Mat(matrix.rows(), matrix.cols(), CV_8UC1);

    for (int i = 0; i < matrix.rows(); i++) {
        for (int j = 0; j < matrix.cols(); j++) {
            uchar pixel = (uchar) std::abs( matrix(i, j) );
            mat.at<uchar>(i, j)  =  pixel;
        }
    }

    //std::cout << "\n\n\n" << matrix << "\n\n\n";

    return mat;
}
