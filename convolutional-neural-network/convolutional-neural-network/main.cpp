#include "convolutional-neural-network.h"
#include "multy-layer-perceptron.h"
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>

// #include "gnuplot-include.h" // Plotting gráfico dos resultados



int main(int argc, const char** argv)
{
	// Gnuplot gnuplot;
	// gnuplot.OutFile("..\\..\\.resources\\gnuplot-output\\res.dat");

	auto LoadData = [](const std::string& folderPath)->std::vector<CnnData> 
	{
		std::vector<CnnData> set;
		int l = -1;

		for (const auto& entry : std::filesystem::directory_iterator(folderPath))
			if (std::filesystem::is_regular_file(entry.path())) {

				std::string fileName = entry.path().filename().string();
				std::string labelStr = Utils::SplitString(fileName, "_")[0];
				int label = std::stoi(labelStr);

				std::string fullPathName = entry.path().string();
				Eigen::MatrixXd input	 = Utils::ImageToMatrix(cv::imread(fullPathName));

				set.push_back( {input, label} );

				if (label > l) { l = label; std::cout << label << "\n"; }
	
			}

		std::random_device rd;
		std::mt19937 g(rd());
		std::shuffle(set.begin(), set.end(), g);

		return set;
	};

	std::vector<CnnData> trainigSet = LoadData("C:\\Users\\octav\\OneDrive\\Desktop\\strain");

	// std::vector<CnnData> testingSet = LoadData(" ... ");

	Eigen::MatrixXd kernel = Eigen::MatrixXd::Constant(3, 3, 1.0);
	std::cout << "\n\n" << kernel << "\n\n\n";

	std::initializer_list<ProcessLayer> processLayer = 
	{
		ProcessLayer(kernel, 0, new ReLU(), new DontPooling()),
	};


	CNN cnn = CNN(	28, 28,			// Resolução
					processLayer,	// Camadas
					{ 100, 10 },	// Arquitetura MLP
					new Sigmoid(),	// Função de Ativação da MLP
					0.003			// LR
				 );

	cnn.MaxAcceptableError(0.000005);

	int epochCounter = -1;

	cnn.Training(trainigSet,
		1,
		[](int label) -> std::vector<double> {
			std::vector<double> correctOutput = std::vector<double>((size_t)10, 0.0);
			correctOutput[label] = 1.0;
			return correctOutput;
		},
		// [&cnn, &trainigSet, &ephocCounter, &gnuplot]() {
		[&cnn, &trainigSet, &epochCounter]() {

			epochCounter++;

			Eigen::MatrixXd confusionMatrix = Eigen::MatrixXd::Zero(10, 10);
			int totalData = 0;
			int errors = 0;


			for (const auto& entry : std::filesystem::directory_iterator("C:\\Users\\octav\\OneDrive\\Desktop\\strain")) {
				if (std::filesystem::is_regular_file(entry.path())) {

					std::string fileName = entry.path().filename().string();
					std::string labelStr = Utils::SplitString(fileName, "_")[0];
					int label = std::stoi(labelStr);

					std::string fullPathName = entry.path().string();
					Eigen::MatrixXd input = Utils::ImageToMatrix(cv::imread(fullPathName));


					std::vector<double> givenOutput = cnn.ProcessInput(input);

					auto it = std::max_element(givenOutput.begin(), givenOutput.end());
					int givenLabel = std::distance(givenOutput.begin(), it);

					confusionMatrix(givenLabel, label) += 1.0;

					totalData++;

					if (givenLabel != label) { errors++; }
				}
			}

			double accuracy = 1.0 - ((double)errors / totalData);
			double kernelMean = cnn.ProcessLayers()[0].Gradient().cwiseProduct(Eigen::MatrixXd::Ones(3, 3)).sum() / 9.0;

			if (epochCounter % 10 == 0) {
				std::cout << "accuracy: " << accuracy << "\n\n";
				std::cout << confusionMatrix << "\n\n";

				std::cout << "gradient:\n" << cnn.ProcessLayers()[0].Gradient() << "\n\n";
				std::cout << "kernel:\n" << cnn.ProcessLayers()[0].Kernel() << "\n\n";
				std::cout << "-------------------------------------\n\n\n";
			}

			// gnuplot.out << ephocCounter << " " << accuracy << " " << kernelMean <<  "\n";
		}
	);

	// gnuplot.out.close();

	Eigen::MatrixXd confusionMatrix = Eigen::MatrixXd::Zero(10, 10);
	int totalData = 0;
	int errors = 0;


	for (const auto& entry : std::filesystem::directory_iterator("C:\\Users\\octav\\OneDrive\\Desktop\\strain"))
		if (std::filesystem::is_regular_file(entry.path())) {

			std::string fileName = entry.path().filename().string();
			std::string labelStr = Utils::SplitString(fileName, "_")[0];
			int label = std::stoi(labelStr);

			std::string fullPathName = entry.path().string();
			Eigen::MatrixXd input = Utils::ImageToMatrix(cv::imread(fullPathName));


			std::vector<double> givenOutput = cnn.ProcessInput(input);

			auto it = std::max_element(givenOutput.begin(), givenOutput.end());
			int givenLabel = std::distance(givenOutput.begin(), it);

			confusionMatrix(givenLabel, label) += 1.0;

			totalData++;

			if (givenLabel != label) { errors++; }
		}

	std::cout << "\n\n\n\n\n\n";

	double accuracy = 1.0 - ((double)errors / totalData);
	std::cout << "accuracy: " << accuracy << "\n\n";
	std::cout << confusionMatrix << "\n\n";


	/*
	gnuplot << "set xrange [0:] \n";
	gnuplot << "plot \'..\\..\\.resources\\gnuplot-output\\res.dat\' using 1:2 w l, ";
	gnuplot << "\'..\\..\\.resources\\gnuplot-output\\res.dat\' using 1:3 w l \n";

	// gnuplot << "set terminal pngcairo enhanced \n set output \'..\\..\\.resources\\gnuplot-output\\accuracy.png\' \n";
	*/

	std::cout << "\n\n\n[SUCESSO - 1]\n\n\n";
	cv::waitKey(0);
}