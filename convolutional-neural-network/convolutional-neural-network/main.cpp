#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <algorithm>
#include <random>
#include "multy-layer-perceptron.h"
#include "convolutional-neural-network.h"




int main(int argc, const char** argv)
{
    /*
    cv::Mat image  =  cv::imread("..\\..\\.resources\\train-debug\\0_1.jpg", cv::IMREAD_GRAYSCALE);
    //cv::resize(image, image, cv::Size(image.rows/2, image.cols/2));

    Eigen::MatrixXd matImg  =  Utils::ImageToMatrix(image);


    cv::namedWindow("Original Imagem", cv::WINDOW_NORMAL);
    cv::imshow("Original Imagem", image);




    Eigen::MatrixXd kernel = Eigen::MatrixXd(3,3);

    kernel <<
        2.04057, 0.590347, 0.0413314,
        0.359109, 0.533219, -0.765308,
        -0.238768, -1.00479, -1.55571;

    // std::vector<double> flatKernel = Utils::FlatMatrix(kernel);
    // std::vector<double> normKernel = Utils::BatchNormalization(flatKernel);
    // kernel = Utils::ReshapeMatrix(normKernel, kernel.rows(), kernel.cols());

    // std::cout << kernel << "\n\n";


    ProcessLayer convolutionLayer = ProcessLayer(kernel, 0 , new ReLU(), new MaxPooling(2, 2));


    /// comvolution
    auto matrix  =  convolutionLayer.CalculateConvolution( matImg );
    std::cout << "\n\n" << matrix << "\n\n";
    auto convolatedImage  =  Utils::MatrixToImage(matrix);

    cv::namedWindow("convulated", cv::WINDOW_NORMAL);
    cv::imshow("convulated", convolatedImage);


    /// relu
    auto matrixWithActFun  =  convolutionLayer.ApplayActivationFunction(matrix);
    auto activatedImage  =  Utils::MatrixToImage(matrixWithActFun);

    cv::namedWindow("activation function", cv::WINDOW_NORMAL);
    cv::imshow("activation function", activatedImage);



    /// pooling
    auto matrixWithPooling  =  convolutionLayer.CalculatePooling(matrixWithActFun);
    auto pooledImage  =  Utils::MatrixToImage(matrixWithPooling);

    cv::namedWindow("pooling", cv::WINDOW_NORMAL);
    cv::imshow("pooling", pooledImage);
    */

    /*
    cv::Mat image  =  cv::imread("..\\..\\.resources\\train-debug\\humano.png", cv::IMREAD_GRAYSCALE);
    Eigen::MatrixXd input  =  Utils::ImageToMatrix(image);


    auto LoadData = [](const std::string& folderPath)->std::vector<CnnData> {
        std::vector<CnnData> set;

        for (const auto& entry : std::filesystem::directory_iterator(folderPath)) {
            if (std::filesystem::is_regular_file(entry.path())) {

                std::string fileName = entry.path().filename().string();
                std::string labelStr = Utils::SplitString(fileName, "_")[0];
                int label = std::stoi(labelStr);

                std::string fullPathName = entry.path().string();
                Eigen::MatrixXd input = Utils::ImageToMatrix( cv::imread(fullPathName) );

                set.push_back( {input, label} );
            }
        }

        return set;
    };

    std::vector<CnnData> Set = LoadData("..\\..\\.resources\\train-debug");

    //std::vector<CnnData> testingSet ={ }; //


    std::vector<DATA> trainigSet;

    for (auto& s : Set) {
        std::vector<double> inputs = Utils::FlatMatrix(s.inputs);
        trainigSet.push_back( {inputs, s.labelIndex } );
    }

    Eigen::MatrixXd kernel  =  Eigen::MatrixXd::Ones(3, 3);

    MLP mlp = MLP(28*28, { 50, 10 }, new Tanh (), 0.03);

    mlp.Training(trainigSet,
                 10000,
                 [](int label) -> std::vector<double> {
                     std::vector<double> correctOutput = std::vector<double>((size_t)10, 0.0);
                     correctOutput[label] = 1.0;
                     return correctOutput;
                 },
                 [&mlp, &trainigSet]() {
                     std::cout << mlp << "\n";

                     size_t index = 0;

                     std::vector<double> givenOutput = mlp.Forward(trainigSet[index].inputs);

                     auto it = std::max_element(givenOutput.begin(), givenOutput.end());
                     int givenLabel = std::distance(givenOutput.begin(), it);

                     std::cout << "corrent output: " << trainigSet[index].labelIndex << "\n";
                     std::cout << "given output:   " << givenLabel << "\n";
                     std::cout << "given vector: \n";
                     for (auto o : givenOutput) { std::cout << o << "  "; }

                     std::cout << "\n\n";
                 }
    );
    */





   


    auto LoadData = [](const std::string& folderPath)->std::vector<CnnData> {
        std::vector<CnnData> set;

        for (const auto& entry : std::filesystem::directory_iterator(folderPath)) {
            if (std::filesystem::is_regular_file(entry.path())) {

                std::string fileName = entry.path().filename().string();
                std::string labelStr = Utils::SplitString(fileName, "_")[0];
                int label = std::stoi(labelStr);

                std::string fullPathName = entry.path().string();
                Eigen::MatrixXd input = Utils::ImageToMatrix( cv::imread(fullPathName) );

                set.push_back( {input, label} );
            }
        }

        std::random_device rd;
        std::mt19937 g(rd());

        std::shuffle(set.begin(), set.end(), g);
        for (auto o : set) { std::cout << o.labelIndex << "  "; }

        return set;
    };

    std::vector<CnnData> trainigSet = LoadData("..\\..\\.resources\\train-debug");

    //std::vector<CnnData> testingSet = { }; //

    Eigen::MatrixXd kernel  =  Eigen::MatrixXd::Ones(3, 3);


    std::initializer_list<ProcessLayer> processLayer ={
        ProcessLayer(kernel, 0, new ReLU(), new MaxPooling(2,2)),
        //ProcessLayer(kernel, 0, new ReLU(), new MaxPooling(2,2))
    };


    CNN cnn = CNN(28, 28, processLayer, { 50, 10 }, new Tanh (), 0.03);

    cnn.MaxAcceptableError(0.005);

    cnn.Training(trainigSet,
                 10,
                 [](int label) -> std::vector<double> {
                     std::vector<double> correctOutput = std::vector<double>((size_t)10, 0.0);
                     correctOutput[label] = 1.0;
                     return correctOutput;
                 },
                 [&cnn, &trainigSet]() {
                     std::cout << cnn << "\n";

                     size_t index = 0;
                     std::vector<double> givenOutput = cnn.ProcessInput(trainigSet[index].inputs);

                     auto it = std::max_element(givenOutput.begin(), givenOutput.end());
                     int givenLabel = std::distance(givenOutput.begin(), it);


                     std::cout << "corrent output: " << trainigSet[index].labelIndex << "\n";
                     std::cout << "given output:   " << givenLabel << "\n";
                     std::cout << "given vector: \n";
                     for (auto o : givenOutput) { std::cout << o << "  "; }

                     std::cout << "\n\n";
                 }
    );
    

    Eigen::MatrixXd confusionMatrix = Eigen::MatrixXd::Zero(10, 10);
    int totalData = 0;
    int errors = 0;


    for (const auto& entry : std::filesystem::directory_iterator("..\\..\\.resources\\train-debug")) {
        if (std::filesystem::is_regular_file(entry.path())) {

            std::string fileName = entry.path().filename().string();
            std::string labelStr = Utils::SplitString(fileName, "_")[0];
            int label = std::stoi(labelStr);

            std::string fullPathName = entry.path().string();
            Eigen::MatrixXd input = Utils::ImageToMatrix(cv::imread(fullPathName));


            std::vector<double> givenOutput = cnn.ProcessInput( input );

            auto it = std::max_element(givenOutput.begin(), givenOutput.end());
            int givenLabel = std::distance(givenOutput.begin(), it);

            confusionMatrix(givenLabel, label) += 1.0;

            totalData++;

            if (givenLabel != label) { errors++; }
        }
    }


    std::cout << "\n\n\n\n\n\n";

    double accuracy = 1.0 - ((double)errors/totalData);
    std::cout << "accuracy: " << accuracy << "\n\n";
    std::cout << confusionMatrix << "\n\n";




    cv::waitKey(0);
    std::cout << "\n\n\n[SUCESSO - 1]\n\n\n";
 
}

