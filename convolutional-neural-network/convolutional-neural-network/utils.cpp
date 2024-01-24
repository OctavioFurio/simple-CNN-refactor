#include "utils.h"



Eigen::MatrixXd Utils::ImageToMatrix(cv::Mat mat)
{
    if (mat.channels() > 1) { cv::cvtColor(mat, mat, cv::COLOR_BGR2GRAY); }

    Eigen::MatrixXd matrix = Eigen::MatrixXd(mat.rows, mat.cols);

    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols; j++) {
            uchar pixel  =  mat.at<uchar>(i, j);
            matrix(i, j) =  (double)pixel;
        }
    }

    return matrix;
}



cv::Mat Utils::MatrixToImage(Eigen::MatrixXd matrix)
{
    cv::Mat mat  =  cv::Mat(matrix.rows(), matrix.cols(), CV_8UC1);

    for (int i = 0; i < matrix.rows(); i++) {
        for (int j = 0; j < matrix.cols(); j++) {
            uchar pixel = (uchar)std::abs(matrix(i, j));
            mat.at<uchar>(i, j)  =  pixel;
        }
    }

    //std::cout << "\n\n\n" << matrix << "\n\n\n";

    return mat;
}
