#include "utils.h"



Eigen::MatrixXd Utils::ImageToMatrix(cv::Mat mat)
{
    if (mat.channels() > 1) { cv::cvtColor(mat, mat, cv::COLOR_BGR2GRAY); }

    Eigen::MatrixXd matrix = Eigen::MatrixXd(mat.rows, mat.cols);

    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols; j++) {
            double pixel  =  (double) mat.at<uchar>(i, j) / 255.0;
            matrix(i, j) =  pixel;
        }
    }

    return matrix;
}



cv::Mat Utils::MatrixToImage(Eigen::MatrixXd matrix)
{
    cv::Mat mat  =  cv::Mat(matrix.rows(), matrix.cols(), CV_8UC1);

    for (int i = 0; i < matrix.rows(); i++) {
        for (int j = 0; j < matrix.cols(); j++) {
            uchar pixel = (uchar) (std::abs(matrix(i, j)) * 255.0);
            mat.at<uchar>(i, j)  =  pixel;

        }
    }

    //std::cout << "\n\n\n" << matrix << "\n\n\n";

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

    double variance = std::sqrt( somaQuadradosDiferencas / values.size() );

    if (variance == 0.0) { variance = std::numeric_limits<double>::min(); }

    return variance;
}



std::vector<double> Utils::BatchNormalization(std::vector<double> originalInput)
{
    double mean =  Utils::Mean(originalInput);
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
    *a  =  Eigen::MatrixXd::Zero(b.rows(), b.cols());

    for (int i = 0; i < b.rows(); i++) {
        for (int j = 0; j < b.cols(); j++) {
            (*a)(i, j)  =  b(i, j);
        }
    }
}
