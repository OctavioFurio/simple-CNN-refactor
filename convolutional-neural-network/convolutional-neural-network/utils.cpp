#include "utils.h"

Eigen::MatrixXd Utils::ImageToMatrix(cv::Mat mat)
{
	if (mat.channels() > 1) { cv::cvtColor(mat, mat, cv::COLOR_BGR2GRAY); }

	Eigen::MatrixXd matrix = Eigen::MatrixXd(mat.rows, mat.cols);

	for (int i = 0; i < mat.rows; i++)
		for (int j = 0; j < mat.cols; j++) {
			double pixel = (double)mat.at<uchar>(i, j) / 255.0;
			matrix(i, j) = pixel;
		}

	return matrix;
}



cv::Mat Utils::MatrixToImage(Eigen::MatrixXd matrix)
{
	cv::Mat mat = cv::Mat(matrix.rows(), matrix.cols(), CV_8UC1);

	for (int i = 0; i < matrix.rows(); i++)
		for (int j = 0; j < matrix.cols(); j++) {
			uchar pixel = (uchar)(std::abs(matrix(i, j)) * 255.0);
			mat.at<uchar>(i, j) = pixel;
		}

	return mat;
}



std::vector<double> Utils::NormalizeVector(const std::vector<double> originalVector, double min, double max)
{
	double minValue = *std::min_element(originalVector.begin(), originalVector.end());
	double maxValue = *std::max_element(originalVector.begin(), originalVector.end());

	std::vector<double> normalizedVector;

	for (double value : originalVector) {
		double normalizedValue = ((value - minValue) / (maxValue - minValue)) * (max - min) + min;
		normalizedVector.push_back(normalizedValue);
	}

	return normalizedVector;
}



double Utils::Mean(std::vector<double> values)
{
	double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
	return mean;
}



double Utils::Variance(std::vector<double> values, double mean)
{
	double somaQuadradosDiferencas = std::accumulate(values.begin(), values.end(), 0.0,
		[mean](double acumulador, double valor) {
			double diferenca = valor - mean;
			return acumulador + diferenca * diferenca;
		});

	double variance = std::sqrt(somaQuadradosDiferencas / values.size());

	// Variância = 0 causa instabilidade numérica. Usemos um Épsilon arbitrariamente pequeno ao invés de zero.
	return variance == 0.0 ? std::numeric_limits<double>::min() : variance;
}



std::vector<double> Utils::BatchNormalization(std::vector<double> originalInput)
{
	double mean = Utils::Mean(originalInput);
	double variance = Utils::Variance(originalInput, mean);

	for (auto& input : originalInput) {
		input = (input - mean) / variance;
	}

	return originalInput;
}



Eigen::MatrixXd Utils::FlipMatrixBy180Degree(Eigen::MatrixXd matrix)
{
	Eigen::MatrixXd flippedMatrix = matrix.colwise().reverse().rowwise().reverse();
	return flippedMatrix;
}




void Utils::StoreMatrixByForce(Eigen::MatrixXd* a, Eigen::MatrixXd b)
{
	*a = Eigen::MatrixXd::Zero(b.rows(), b.cols());

	for (int i = 0; i < b.rows(); i++) {
		for (int j = 0; j < b.cols(); j++) {
			(*a)(i, j) = b(i, j);
		}
	}
}



std::vector<std::string> Utils::SplitString(const std::string& input, const std::string& delimiter)
{
	std::vector<std::string> result;
	size_t start = 0;
	size_t end = input.find(delimiter);

	while (end != std::string::npos) {
		result.push_back(input.substr(start, end - start));
		start = end + delimiter.length();
		end = input.find(delimiter, start);
	}

	result.push_back(input.substr(start, end));  // Handle the last token

	return result;
}



Eigen::MatrixXd Utils::ReshapeMatrix(std::vector<double> gradients, size_t rows, size_t cols)
{
	Eigen::MatrixXd reshapedMatrix = Eigen::MatrixXd::Zero(rows, cols);

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			reshapedMatrix(i, j) = gradients[i * cols + j];
		}
	}

	return reshapedMatrix;
}



std::vector<double> Utils::FlatMatrix(Eigen::MatrixXd input)
{
	std::vector<double> vec;
	vec.reserve(input.size());

	for (int i = 0; i < input.rows(); i++) {
		for (int j = 0; j < input.cols(); j++) {
			vec.push_back(input(i, j));
		}
	}

	return vec;
}



Eigen::MatrixXd Utils::BatchNormalization(Eigen::MatrixXd mat)
{
	std::vector<double> flattedMatrix = Utils::FlatMatrix(mat);
	std::vector<double> normalizedMatrixarr = Utils::BatchNormalization(flattedMatrix);
	Eigen::MatrixXd normalizedMatrix = Utils::ReshapeMatrix(normalizedMatrixarr, mat.rows(), mat.cols());

	return normalizedMatrix;
}