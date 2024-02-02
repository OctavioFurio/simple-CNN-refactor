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
    cv::Mat image  =  cv::imread("..\\..\\.resources\\humano-original.png", cv::IMREAD_GRAYSCALE);
    //cv::resize(image, image, cv::Size(image.rows/2, image.cols/2));

    Eigen::MatrixXd matImg  =  Utils::ImageToMatrix(image);


    cv::namedWindow("Original Imagem", cv::WINDOW_NORMAL);
    cv::imshow("Original Imagem", image);


    ProcessLayer convolutionLayer = ProcessLayer(Kernel3x3::Identity(), 0, new ReLU(), new MaxPooling(2, 2));
    auto matrix  =  convolutionLayer.CalculateConvolution( matImg );
    auto convolatedImage  =  Utils::MatrixToImage(matrix);
    cv::namedWindow("Identity", cv::WINDOW_NORMAL);
    cv::imshow("Identity", convolatedImage);


    convolutionLayer = ProcessLayer(Kernel3x3::AverageBlur(), 0, new ReLU(), new MaxPooling(2, 2));
    matrix  =  convolutionLayer.CalculateConvolution(matImg);
    convolatedImage  =  Utils::MatrixToImage(matrix);
    cv::namedWindow("AverageBlur", cv::WINDOW_NORMAL);
    cv::imshow("AverageBlur", convolatedImage);


    convolutionLayer = ProcessLayer(Kernel3x3::EdgeEnhancement(), 0, new ReLU(), new MaxPooling(2, 2));
    matrix  =  convolutionLayer.CalculateConvolution(matImg);
    convolatedImage  =  Utils::MatrixToImage(matrix);
    cv::namedWindow("EdgeEnhancement", cv::WINDOW_NORMAL);
    cv::imshow("EdgeEnhancement", convolatedImage);


    convolutionLayer = ProcessLayer(Kernel3x3::HorizontalLineDetection(), 0, new ReLU(), new MaxPooling(2, 2));
    matrix  =  convolutionLayer.CalculateConvolution(matImg);
    convolatedImage  =  Utils::MatrixToImage(matrix);
    cv::namedWindow("HorizontalLineDetection", cv::WINDOW_NORMAL);
    cv::imshow("HorizontalLineDetection", convolatedImage);


    convolutionLayer = ProcessLayer(Kernel3x3::Laplacian(), 0, new ReLU(), new MaxPooling(2, 2));
    matrix  =  convolutionLayer.CalculateConvolution(matImg);
    convolatedImage  =  Utils::MatrixToImage(matrix);
    cv::namedWindow("Laplacian", cv::WINDOW_NORMAL);
    cv::imshow("Laplacian", convolatedImage);


    convolutionLayer = ProcessLayer(Kernel3x3::VerticalLineDetection(), 0, new ReLU(), new MaxPooling(2, 2));
    matrix  =  convolutionLayer.CalculateConvolution(matImg);
    convolatedImage  =  Utils::MatrixToImage(matrix);
    cv::namedWindow("VerticalLineDetection", cv::WINDOW_NORMAL);
    cv::imshow("VerticalLineDetection", convolatedImage);




    convolutionLayer = ProcessLayer(Kernel3x3::KrischCompassEast(), 0, new ReLU(), new MaxPooling(2, 2));
    matrix  =  convolutionLayer.CalculateConvolution(matImg);
    convolatedImage  =  Utils::MatrixToImage(matrix);
    cv::namedWindow("KrischCompassEast", cv::WINDOW_NORMAL);
    cv::imshow("KrischCompassEast", convolatedImage);
  
    convolutionLayer = ProcessLayer(Kernel3x3::KrischCompassNorth(), 0, new ReLU(), new MaxPooling(2, 2));
    matrix  =  convolutionLayer.CalculateConvolution(matImg);
    convolatedImage  =  Utils::MatrixToImage(matrix);
    cv::namedWindow("KrischCompassNorth", cv::WINDOW_NORMAL);
    cv::imshow("KrischCompassNorth", convolatedImage);

    convolutionLayer = ProcessLayer(Kernel3x3::KrischCompassNorthEast(), 0, new ReLU(), new MaxPooling(2, 2));
    matrix  =  convolutionLayer.CalculateConvolution(matImg);
    convolatedImage  =  Utils::MatrixToImage(matrix);
    cv::namedWindow("KrischCompassNorthEast", cv::WINDOW_NORMAL);
    cv::imshow("KrischCompassNorthEast", convolatedImage);

    convolutionLayer = ProcessLayer(Kernel3x3::KrischCompassNorthWest(), 0, new ReLU(), new MaxPooling(2, 2));
    matrix  =  convolutionLayer.CalculateConvolution(matImg);
    convolatedImage  =  Utils::MatrixToImage(matrix);
    cv::namedWindow("KrischCompassNorthWest", cv::WINDOW_NORMAL);
    cv::imshow("KrischCompassNorthWest", convolatedImage);

    convolutionLayer = ProcessLayer(Kernel3x3::KrischCompassWest(), 0, new ReLU(), new MaxPooling(2, 2));
    matrix  =  convolutionLayer.CalculateConvolution(matImg);
    convolatedImage  =  Utils::MatrixToImage(matrix);
    cv::namedWindow("KrischCompassWest", cv::WINDOW_NORMAL);
    cv::imshow("KrischCompassWest", convolatedImage);

    convolutionLayer = ProcessLayer(Kernel3x3::KrischCompassSouth(), 0, new ReLU(), new MaxPooling(2, 2));
    matrix  =  convolutionLayer.CalculateConvolution(matImg);
    convolatedImage  =  Utils::MatrixToImage(matrix);
    cv::namedWindow("KrischCompassSouth", cv::WINDOW_NORMAL);
    cv::imshow("KrischCompassSouth", convolatedImage);

    convolutionLayer = ProcessLayer(Kernel3x3::KrischCompassSouthEast(), 0, new ReLU(), new MaxPooling(2, 2));
    matrix  =  convolutionLayer.CalculateConvolution(matImg);
    convolatedImage  =  Utils::MatrixToImage(matrix);
    cv::namedWindow("KrischCompassNorthEast", cv::WINDOW_NORMAL);
    cv::imshow("KrischCompassNorthEast", convolatedImage);

    convolutionLayer = ProcessLayer(Kernel3x3::KrischCompassSouthWest(), 0, new ReLU(), new MaxPooling(2, 2));
    matrix  =  convolutionLayer.CalculateConvolution(matImg);
    convolatedImage  =  Utils::MatrixToImage(matrix);
    cv::namedWindow("KrischCompassNorthEast", cv::WINDOW_NORMAL);
    cv::imshow("KrischCompassNorthEast", convolatedImage);





    convolutionLayer = ProcessLayer(Kernel3x3::RobinsonCompassEast(), 0, new ReLU(), new MaxPooling(2, 2));
    matrix  =  convolutionLayer.CalculateConvolution(matImg);
    convolatedImage  =  Utils::MatrixToImage(matrix);
    cv::namedWindow("RobinsonCompassEast", cv::WINDOW_NORMAL);
    cv::imshow("RobinsonCompassEast", convolatedImage);

    convolutionLayer = ProcessLayer(Kernel3x3::RobinsonCompassNorth(), 0, new ReLU(), new MaxPooling(2, 2));
    matrix  =  convolutionLayer.CalculateConvolution(matImg);
    convolatedImage  =  Utils::MatrixToImage(matrix);
    cv::namedWindow("RobinsonCompassNorth", cv::WINDOW_NORMAL);
    cv::imshow("RobinsonCompassNorth", convolatedImage);

    convolutionLayer = ProcessLayer(Kernel3x3::RobinsonCompassNorthEast(), 0, new ReLU(), new MaxPooling(2, 2));
    matrix  =  convolutionLayer.CalculateConvolution(matImg);
    convolatedImage  =  Utils::MatrixToImage(matrix);
    cv::namedWindow("RobinsonCompassNorthEast", cv::WINDOW_NORMAL);
    cv::imshow("RobinsonCompassNorthEast", convolatedImage);

    convolutionLayer = ProcessLayer(Kernel3x3::RobinsonCompassNorthWest(), 0, new ReLU(), new MaxPooling(2, 2));
    matrix  =  convolutionLayer.CalculateConvolution(matImg);
    convolatedImage  =  Utils::MatrixToImage(matrix);
    cv::namedWindow("RobinsonCompassNorthWest", cv::WINDOW_NORMAL);
    cv::imshow("RobinsonCompassNorthWest", convolatedImage);

    convolutionLayer = ProcessLayer(Kernel3x3::RobinsonCompassWest(), 0, new ReLU(), new MaxPooling(2, 2));
    matrix  =  convolutionLayer.CalculateConvolution(matImg);
    convolatedImage  =  Utils::MatrixToImage(matrix);
    cv::namedWindow("RobinsonCompassWest", cv::WINDOW_NORMAL);
    cv::imshow("RobinsonCompassWest", convolatedImage);

    convolutionLayer = ProcessLayer(Kernel3x3::RobinsonCompassSouth(), 0, new ReLU(), new MaxPooling(2, 2));
    matrix  =  convolutionLayer.CalculateConvolution(matImg);
    convolatedImage  =  Utils::MatrixToImage(matrix);
    cv::namedWindow("RobinsonCompassSouth", cv::WINDOW_NORMAL);
    cv::imshow("RobinsonCompassSouth", convolatedImage);

    convolutionLayer = ProcessLayer(Kernel3x3::RobinsonCompassSouthEast(), 0, new ReLU(), new MaxPooling(2, 2));
    matrix  =  convolutionLayer.CalculateConvolution(matImg);
    convolatedImage  =  Utils::MatrixToImage(matrix);
    cv::namedWindow("RobinsonCompassNorthEast", cv::WINDOW_NORMAL);
    cv::imshow("RobinsonCompassNorthEast", convolatedImage);

    convolutionLayer = ProcessLayer(Kernel3x3::RobinsonCompassSouthWest(), 0, new ReLU(), new MaxPooling(2, 2));
    matrix  =  convolutionLayer.CalculateConvolution(matImg);
    convolatedImage  =  Utils::MatrixToImage(matrix);
    cv::namedWindow("RobinsonCompassNorthEast", cv::WINDOW_NORMAL);
    cv::imshow("RobinsonCompassNorthEast", convolatedImage);
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
        inputs.insert(inputs.begin(), 1.0);
        trainigSet.push_back( {inputs, s.labelIndex } );
    }

    Eigen::MatrixXd kernel  =  Eigen::MatrixXd::Ones(3, 3);

    MLP mlp = MLP(28*28, { 70, 50, 10 }, new Tanh(), 0.03);

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

                     std::cout << "\n\nt1";
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


            std::vector<double> inputs = Utils::FlatMatrix(input);
            inputs.insert(inputs.begin(), 1.0);


            std::vector<double> givenOutput = mlp.Forward(inputs);

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
    */


    
    auto LoadData = [](const std::string& folderPath)->std::vector<CnnData> {
        std::vector<CnnData> set;

        int l = -1;

        for (const auto& entry : std::filesystem::directory_iterator(folderPath)) {
            if (std::filesystem::is_regular_file(entry.path())) {

                std::string fileName = entry.path().filename().string();
                std::string labelStr = Utils::SplitString(fileName, "_")[0];
                int label = std::stoi(labelStr);

                std::string fullPathName = entry.path().string();
                Eigen::MatrixXd input = Utils::ImageToMatrix( cv::imread(fullPathName) );

                set.push_back( {input, label} );
                
                if (label > l) { l = label; std::cout << label << "\n"; }

            }
        }

        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(set.begin(), set.end(), g);

        return set;
    };

    std::vector<CnnData> trainigSet = LoadData("..\\..\\.resources\\train-debug");

    //std::vector<CnnData> testingSet = { }; //

    Eigen::MatrixXd kernel = Eigen::MatrixXd::Constant(7,7, 1.0/49.0);

    std::cout << "\n\n" << kernel << "\n\n\n";


    std::initializer_list<ProcessLayer> processLayer ={
        ProcessLayer(kernel, 0, new ReLU(), new MaxPooling(3,3)),
        //ProcessLayer(kernel, 0, new ReLU(), new MaxPooling(2,2))
    };


    CNN cnn = CNN(28, 28, processLayer, { 50, 10 }, new Tanh (), 0.03);

    cnn.MaxAcceptableError(0.05);
    cnn.LearningRate(0.003);

    cnn.Training(trainigSet,
                 100,
                 [](int label) -> std::vector<double> {
                     std::vector<double> correctOutput = std::vector<double>((size_t)10, 0.0);
                     correctOutput[label] = 1.0;
                     return correctOutput;
                 },
                 [&cnn, &trainigSet]() {
                     std::cout << "\n-------------------------------------------------------\n\n";
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

