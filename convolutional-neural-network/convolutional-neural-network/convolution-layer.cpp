#include "convolution-layer.h"




ConvolutionLayer::ConvolutionLayer(Eigen::MatrixXd kernel, int padding)
    : _kernel(kernel), _padding(padding)
{
    _gradient = Eigen::MatrixXd::Zero(_kernel.rows(), _kernel.cols());
    _bias = 1.0;
}



ConvolutionLayer::ConvolutionLayer(std::vector<std::vector<double>> kernel, int padding)
    : _padding(padding)
{
    _bias = 1.0;

    int kernelRows = kernel.size();
    int kernelCols = kernel[0].size();

    _kernel = Eigen::MatrixXd(kernelRows, kernelCols);

    for (auto row : kernel) {
        assert(kernelCols == row.size());
        for (auto col : row) {
            _kernel << col;
        }
    }

}



ConvolutionLayer::~ConvolutionLayer()
{
}



Eigen::MatrixXd ConvolutionLayer::CalculateConvolution(Eigen::MatrixXd input)
{
    Eigen::MatrixXd convolutedInput;

    if (_padding >= 1) {
        convolutedInput  =  Convolution2D(input, _kernel, _padding);
    } else {
        convolutedInput  =  Convolution2D(input, _kernel);
    }

    return convolutedInput;
}



Eigen::MatrixXd ConvolutionLayer::CalculateConvolution(cv::Mat image)
{
    Eigen::MatrixXd input  = Utils::ImageToMatrix(image);

    Eigen::MatrixXd convolutedInput;

    if (_padding >= 1) {
        convolutedInput  =  Convolution2D(input, _kernel, _padding);
    } else {
        convolutedInput  =  Convolution2D(input, _kernel);
    }

    return convolutedInput;
}



Eigen::MatrixXd ConvolutionLayer::Backward(Eigen::MatrixXd input, Eigen::MatrixXd incomeGradient, double learningRate)
{
    // calcular o dE/dX (derivada parcial da funcao custo com relacao a entrada) usado para atualizar a proxima camada
    Eigen::MatrixXd flippedMatrix  =  Utils::FlipMatrixBy180Degree(_kernel);
    Eigen::MatrixXd gradientOfLostWithRespctToInput  =  Convolution2D(incomeGradient, flippedMatrix, flippedMatrix.rows()-1, flippedMatrix.cols()-1);




    // update kernel    (dE/dF)     updated after gradient Of Lost With Respct To Input was calculated
    _gradient = Convolution2D(input, incomeGradient);
    _kernel = _kernel - learningRate * _gradient;


    //_kernel = Utils::BatchNormalization( _kernel );
    
   


    return gradientOfLostWithRespctToInput;
}



Eigen::MatrixXd ConvolutionLayer::Convolution2D(Eigen::MatrixXd input, Eigen::MatrixXd kernel)
{
    //std::cout << "conv input:\n" << input << "\n\n";
    //std::cout << "conv kernel:\n" << kernel << "\n\n";


    const int kernelRows = kernel.rows();
    const int kernelCols = kernel.cols();
    const int rows = (input.rows() - kernelRows) + 1;
    const int cols = (input.cols() - kernelCols) + 1;

    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(rows, cols);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double sum = input.block(i, j, kernelRows, kernelCols).cwiseProduct(kernel).sum();
            result(i, j) = sum;

            //std::cout << "\n\n" << result << "\n\n";
        }
    }

    //std::cout << "\n\n" << result << "\n\n";

    return result;
}



Eigen::MatrixXd ConvolutionLayer::Convolution2D(Eigen::MatrixXd input, Eigen::MatrixXd kernel, int padding)
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



Eigen::MatrixXd& ConvolutionLayer::Convolution2D(Eigen::MatrixXd input, Eigen::MatrixXd kernel, int rowPadding, int colPadding)
{
    int kernelRows = kernel.rows();
    int kernelCols = kernel.cols();
    int rows = input.rows() - kernelRows + 2*rowPadding + 1;
    int cols = input.cols() - kernelCols + 2*colPadding + 1;

    Matrix padded = Matrix::Zero(input.rows() + 2*rowPadding, input.cols() + 2*colPadding);
    padded.block(rowPadding, colPadding, input.rows(), input.cols()) = input;

    Matrix result = Matrix::Zero(rows, cols);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double sum = padded.block(i, j, kernelRows, kernelCols).cwiseProduct(kernel).sum();
            result(i, j) = sum;
        }
    }

    return result;
}



Eigen::MatrixXd ConvolutionLayer::Convolution2DWithVisualization(Eigen::MatrixXd& input, Eigen::MatrixXd& kernel)
{
    const int kernelRows = kernel.rows();
    const int kernelCols = kernel.cols();
    const int rows = (input.rows() - kernelRows) + 1;
    const int cols = (input.cols() - kernelCols) + 1;

    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(rows, cols);
    cv::Mat mat  =  cv::Mat(rows, cols, CV_8UC1);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double sum = input.block(i, j, kernelRows, kernelCols).cwiseProduct(kernel).sum();
            result(i, j) = sum;
             
            mat.at<uchar>(i, j)  =  (uchar)(sum * 255);

            cv::namedWindow("teste-1", cv::WINDOW_NORMAL);
            cv::imshow("teste-1", mat);
            cv::waitKey(2);
        }
    }

    return result;
}



Eigen::MatrixXd ConvolutionLayer::Kernel()
{
    return _kernel;
}



Eigen::MatrixXd ConvolutionLayer::Gradient()
{
    return _gradient;
}

void ConvolutionLayer::NormalizeKernel()
{
    Eigen::MatrixXd normalizedKernel = Utils::BatchNormalization(_kernel);
    _kernel = normalizedKernel;
}
