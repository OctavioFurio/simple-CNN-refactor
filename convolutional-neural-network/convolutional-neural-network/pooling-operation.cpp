#include "pooling-operation.h"



/// -----------------
/// Max Pooling
/// -----------------

MaxPooling::MaxPooling(unsigned rows, unsigned cols)
	: _rows(rows), _cols(cols)
{
    _unitaryMatrix  =  Eigen::MatrixXd::Ones(_rows, _cols);
}



Matrix MaxPooling::Pooling(Eigen::MatrixXd& input)
{
    const int rows = (input.rows() - _rows) + 1;
    const int cols = (input.cols() - _cols) + 1;

    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(rows, cols);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double sum = input.block(i, j, _rows, _cols).cwiseProduct(_unitaryMatrix).sum();
            result(i, j)  =  sum / (double)(_rows * _cols);
        }
    }

    return result;
}





/// -----------------
/// Average Pooling
/// -----------------

AveragePooling::AveragePooling(unsigned rows, unsigned cols)
    : _rows(rows), _cols(cols)
{
}



Matrix AveragePooling::Pooling(Eigen::MatrixXd& input)
{
    const int rows = (input.rows() - _rows) + 1;
    const int cols = (input.cols() - _cols) + 1;

    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(rows, cols);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double maxElement = input.block(i, j, _rows, _cols).maxCoeff();
            result(i, j)  =  maxElement;
        }
    }

    return result;
}




/// -----------------
/// Don't Pooling
/// -----------------

DontPooling::DontPooling()
{
}

Matrix DontPooling::Pooling(Eigen::MatrixXd& input)
{
    return input;
}
