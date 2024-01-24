#include "pooling-operation.h"



/// -----------------
/// AveragePooling
/// -----------------

AveragePooling::AveragePooling(size_t rows, size_t cols)
	: _rows(rows), _cols(cols)
{
    _unitaryMatrix  =  Eigen::MatrixXd::Ones(_rows, _cols);
}



Matrix AveragePooling::FowardPooling(Eigen::MatrixXd& input)
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

size_t AveragePooling::Rows()
{
    return _rows;
}

size_t AveragePooling::Cols()
{
    return _cols;
}





/// -----------------
/// MaxPooling
/// -----------------

MaxPooling::MaxPooling(size_t rows, size_t cols)
    : _rows(rows), _cols(cols)
{
}



Matrix MaxPooling::FowardPooling(Eigen::MatrixXd& input)
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

size_t MaxPooling::Rows()
{
    return _rows;
}

size_t MaxPooling::Cols()
{
    return _cols;
}




/// -----------------
/// Don't Pooling
/// -----------------

DontPooling::DontPooling()
{
}

Matrix DontPooling::FowardPooling(Eigen::MatrixXd& input)
{
    return input;
}

size_t DontPooling::Rows()
{
    return 0;
}

size_t DontPooling::Cols()
{
    return 0;
}
