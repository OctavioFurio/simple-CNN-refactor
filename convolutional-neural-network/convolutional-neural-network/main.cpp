#include <iostream>
#include <opencv2/opencv.hpp>
#include "multy-layer-perceptron.h"
#include "convolutional-neural-network.h"




int main(int argc, const char** argv)
{

    cv::Mat image  =  cv::imread("..\\..\\.resources\\humano.png", cv::IMREAD_GRAYSCALE);
    Eigen::MatrixXd input  =  Utils::ImageToMatrix(image);

    // LOAD TRAINIG DATA
    std::vector<CnnData> trainigSet = { }; //
    std::vector<CnnData> testingSet ={ }; //


    Eigen::MatrixXd kernel  =  Eigen::MatrixXd::Ones(3,3);


    std::initializer_list<ProcessLayer> processLayer = { 
        ProcessLayer(kernel, 0, new ReLU(), new MaxPooling(2,2)),
        ProcessLayer(kernel, 0, new ReLU(), new MaxPooling(2,2))
    };


    CNN cnn = CNN(28, 28, processLayer, { 5, 10 }, new Tanh (),  0.01 );
    cnn.Training(trainigSet, 
                 [](int label) -> std::vector<double> {
                     std::vector<double> correctOutput = std::vector<double>((size_t)10, 0.0);
                     correctOutput[label] = 1.0;
                     return correctOutput
                 }
    );

    
    for (auto testInput : testingSet) {
        std::vector<double> givenOutput = cnn.ProcessInput(testInput);
        // parse givenOutput to a index (max of all element fron the vector)

        std::cout << "correct output: " << label << "\n";
        std::cout << "given Output:   " << givenOutput << "\n\n";
    }
    


    std::cout << "\n\n\n[SUCESSO - 1]\n\n\n";
    return 0;
}

