#include <iostream>
#include <opencv2/opencv.hpp>
#include "multy-layer-perceptron.h"
#include "process-layer.h"




int main(int argc, const char** argv)
{
    /*
    Eigen::MatrixXd kernel(3, 3);
    kernel <<
        -1, 0, 1,
        -1, 0, 1,
        -1, 0, 1;

    std::cout << "Kernel:\n" << kernel << "\n\n";

    Eigen::MatrixXd input(6, 6);
    input << 3, 1, 0, 2, 5, 6,
        4, 2, 1, 1, 4, 7,
        5, 4, 0, 0, 1, 2,
        1, 2, 2, 1, 3, 4,
        6, 3, 1, 0, 5, 2,
        3, 1, 0, 1, 3, 3;

    std::cout << "Input:\n" << input << "\n\n";

    auto output = ConvolutionLayer::Convolution2D(input, kernel, 0);
    std::cout << "Convolution:\n" << output << "\n";
    */
	
    
    cv::Mat image  =  cv::imread("..\\..\\.resources\\humano.png", cv::IMREAD_GRAYSCALE);
    //cv::resize(image, image, cv::Size(image.rows/2, image.cols/2));

    cv::namedWindow("Original Imagem", cv::WINDOW_NORMAL);
    cv::imshow("Original Imagem", image);




    Eigen::MatrixXd kernel = Eigen::MatrixXd(3,3);

    kernel <<
        1, 0, -1,
        2, 0, -2,
        1, 0, -1;

    ProcessLayer convolutionLayer = ProcessLayer(kernel, 0, new ReLU(), new MaxPooling(2, 2));



    /// comvolution
    auto matrix  =  convolutionLayer.CalculateConvolution( image );
    auto convolatedImage  =  ProcessLayer::MatrixToImage(matrix);

    cv::namedWindow("convulated", cv::WINDOW_NORMAL);
    cv::imshow("convulated", convolatedImage);



    /// relu
    auto matrixWithActFun  =  convolutionLayer.ApplayActivationFunction(matrix);
    auto activatedImage  =  ProcessLayer::MatrixToImage(matrixWithActFun);

    cv::namedWindow("activation function", cv::WINDOW_NORMAL);
    cv::imshow("activation function", activatedImage);



    /// pooling
    auto matrixWithPooling  =  convolutionLayer.CalculatePooling(matrixWithActFun);
    auto pooledImage  =  ProcessLayer::MatrixToImage(matrixWithPooling);

    cv::namedWindow("pooling", cv::WINDOW_NORMAL);
    cv::imshow("pooling", pooledImage);




    cv::waitKey(0);
    
    
    std::cout << "\n\n\n[SUCESSO]\n\n\n";
}

